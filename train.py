import os
import cv2
import copy
import lpips
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn.utils.spectral_norm as SpectralNorm

from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities import rank_zero_only
from pytorch_fid import fid_score

from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser
from omegaconf import OmegaConf

ImageFile.LOAD_TRUNCATED_IMAGES = True

def weight_init(m, weight_init_type):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if weight_init_type == 'normal':
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif weight_init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight)
        elif weight_init_type == 'xavier':
            nn.init.xavier_uniform_(m.weight, 1.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)
        
class NearestNeighbor2d(nn.Module):
    def __init__(self, scale):
        super(NearestNeighbor2d, self).__init__()
        self.scale = scale
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode='nearest')

def translation(x, ratio=1/8):
    bs, c, h, w = x.shape
    shift_h = int(round(ratio*h))
    shift_w = int(round(ratio*w))

    dh = torch.randint(-shift_h, shift_h, (bs, 1, 1), device=x.device)
    dw = torch.randint(-shift_w, shift_w, (bs, 1, 1), device=x.device)

    grid_b, grid_h, grid_w = torch.meshgrid(
        torch.arange(bs, device=x.device),
        torch.arange(h, device=x.device),
        torch.arange(w, device=x.device)
    )
    grid_h = torch.clamp(grid_h + dh + 1, 0, h+1)
    grid_w = torch.clamp(grid_w + dw + 1, 0, w+1)
    
    x_pad = F.pad(x, [1, 1, 1, 1]).permute(0, 2, 3, 1)
    out = x_pad[grid_b, grid_h, grid_w].permute(0, 3, 1, 2)
    return out

def cutout(x, ratio=1/2):
    bs, c, h, w = x.shape
    cutout_h = int(round(ratio*h))
    cutout_w = int(round(ratio*w))

    dh = torch.randint(0, h-cutout_h%2, (bs, 1, 1), device=x.device)
    dw = torch.randint(0, w-cutout_w%2, (bs, 1, 1), device=x.device)

    grid_b, grid_h, grid_w = torch.meshgrid(
        torch.arange(bs, device=x.device),
        torch.arange(cutout_h, device=x.device),
        torch.arange(cutout_w, device=x.device)
    )
    grid_h = torch.clamp(grid_h + dh - int(cutout_h/2), 0, h-1)
    grid_w = torch.clamp(grid_w + dw - int(cutout_w/2), 0, w-1)
    mask = torch.ones_like(x[:, 0])
    mask[grid_b, grid_h, grid_w] = 0
    return x * mask.unsqueeze(1)

def brightness(x):
    bs, c, h, w = x.shape
    return x + torch.rand(bs, 1, 1, 1, device=x.device) - 0.5

def saturation(x):
    bs, c, h, w = x.shape
    m = x.mean(1, keepdim=True)
    return torch.rand(bs, 1, 1, 1, device=x.device) * 2 * (x - m) + m

def contrast(x):
    bs, c, h, w = x.shape
    m = x.mean((1, 2, 3), keepdim=True)
    return (torch.rand(bs, 1, 1, 1, device=x.device) + 0.5) * (x - m) + m

def crop(x, corner=None):
    # (bs, nc, 16, 16)
    bs, c, h, w = x.shape
    if corner is None:
        corner = random.randint(0, 3)

    if corner == 0:
        return x[..., :h//2, :w//2], corner
    elif corner == 1:
        return x[..., :h//2, w//2:], corner
    elif corner == 2:
        return x[..., h//2:, :w//2], corner
    elif corner == 3:
        return x[..., h//2:, w//2:], corner
    

class DiffAug(nn.Module):
    def __init__(self, policies=['color, translation']):
        super(DiffAug, self).__init__()
        self.policies = policies
        self.diffaugs = {
            'translation': [translation],
            'cutout': [cutout],
            'color': [brightness, saturation, contrast]
        }

    def forward(self, x):
        o = x
        for p in self.policies:
            for da in self.diffaugs[p]:
                o = da(o)
        return o
    
class DCGANGenerator64(nn.Module):
    def __init__(self, oc, nz):
        super(DCGANGenerator64, self).__init__()
        self.nz = nz
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, oc, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z.reshape(-1, self.nz, 1, 1))

class DCGANDiscriminator64(nn.Module):
    def __init__(self):
        super(DCGANDiscriminator64, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1, 4, 1, 0)
        )
    
    def forward(self, x):
        return self.model(x)

class FirstDiscBlk(nn.Module):
    def __init__(self, ic, oc):
        super(FirstDiscBlk, self).__init__()
        self.model = nn.Sequential(
            SpectralNorm(nn.Conv2d(ic, oc, 3, 1, 1)),
            nn.ReLU(),
            SpectralNorm(nn.Conv2d(oc, oc, 3, 1, 1)),
            nn.AvgPool2d(2)
        )
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(nn.Conv2d(ic, oc, 1, 1, 0))
        )
    
    def forward(self, x):
        return self.model(x) + self.shortcut(x)


class DiscBlk(nn.Module):
    def __init__(self, ic, hc, oc, downsample):
        super(DiscBlk, self).__init__()
        if downsample:
            self.avgpool = nn.AvgPool2d(2)
        else:
            self.avgpool = nn.Identity()

        self.model = nn.Sequential(
            nn.ReLU(),
            SpectralNorm(nn.Conv2d(ic, hc, 3, 1, 1)),
            nn.ReLU(),
            SpectralNorm(nn.Conv2d(hc, oc, 3, 1, 1))
        )
        
        if ic != oc or downsample:
            self.shortcut = nn.Sequential(
                SpectralNorm(nn.Conv2d(ic, oc, 1, 1, 0)),
                nn.AvgPool2d(2)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        return self.avgpool(self.model(x)) + self.shortcut(x)

class GenBlk(nn.Module):
    def __init__(self, ic, hc, oc, upsample):
        super(GenBlk, self).__init__()
        if upsample:
            self.nn_up = NearestNeighbor2d(2)
        else:
            self.nn_up = nn.Identity()
        
        self.model = nn.Sequential(
            nn.BatchNorm2d(ic),
            nn.ReLU(),
            self.nn_up,
            nn.Conv2d(ic, hc, 3, 1, 1),
            nn.BatchNorm2d(hc),
            nn.ReLU(),
            nn.Conv2d(hc, oc, 3, 1, 1)
        )
        
        if ic != oc or upsample:
            self.shortcut = nn.Sequential(
                NearestNeighbor2d(2),
                nn.Conv2d(ic, oc, 1, 1, 0)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        return self.model(x) + self.shortcut(x)


class SNGANGenerator32(nn.Module):
    def __init__(self, oc, nz, ngf, bottom):
        super(SNGANGenerator32, self).__init__()
        self.ngf = ngf
        self.bottom = bottom
        self.projection = nn.Linear(nz, bottom * bottom * ngf)
        self.model = nn.Sequential(
            GenBlk(ngf, ngf, ngf, True),
            GenBlk(ngf, ngf, ngf, True),
            GenBlk(ngf, ngf, ngf, True),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            nn.Conv2d(ngf, oc, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        out = self.projection(z).reshape(-1, self.ngf, self.bottom, self.bottom)
        return self.model(out)


class SNGANDiscriminator32(nn.Module):
    def __init__(self, ic, ndf):
        super(SNGANDiscriminator32, self).__init__()
        self.ndf = ndf
        self.model = nn.Sequential(
            FirstDiscBlk(ic, ndf),
            DiscBlk(ndf, ndf, ndf, True),
            DiscBlk(ndf, ndf, ndf, False),
            DiscBlk(ndf, ndf, ndf, False),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = SpectralNorm(nn.Linear(ndf, 1, bias=False))
    
    def forward(self, x):
        out = self.model(x).reshape(-1, self.ndf)
        return self.classifier(out).reshape(-1, 1, 1, 1)


class FastGANGenInitBlk(nn.Module):
    def __init__(self, nz, oc):
        super(FastGANGenInitBlk, self).__init__()
        self.model = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(nz, oc*2, 4, 1, 0, bias=False)),
            nn.BatchNorm2d(oc*2),
        )
    def forward(self, x):
        return F.glu(self.model(x), 1)

    
class FastGANDiscInitBlk(nn.Module):
    def __init__(self, ic, hc, oc, sz):
        super(FastGANDiscInitBlk, self).__init__()
        if sz == 256:
            self.model = nn.Sequential(
                SpectralNorm(nn.Conv2d(ic, oc, 3, 1, 1, bias=False)),
                nn.LeakyReLU(0.2)
            )
        if sz == 512:
            self.model = nn.Sequential(
                SpectralNorm(nn.Conv2d(ic, oc, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2),
            )
        if sz == 1024:
            self.model = nn.Sequential(
                SpectralNorm(nn.Conv2d(ic, hc, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2),
                SpectralNorm(nn.Conv2d(hc, oc, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(oc),
                nn.LeakyReLU(0.2)
            )
    def forward(self, x):
        return self.model(x)
    
    
class FastGANUpBlk(nn.Module):
    def __init__(self, ic, oc):
        super(FastGANUpBlk, self).__init__()
        self.model = nn.Sequential(
            NearestNeighbor2d(2),
            SpectralNorm(nn.Conv2d(ic, oc*2, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(oc*2)
        )
    def forward(self, x):
        return F.glu(self.model(x), 1)
    
    
class FastGANDownBlk(nn.Module):
    def __init__(self, ic, oc):
        super(FastGANDownBlk, self).__init__()
        self.model = nn.Sequential(
            SpectralNorm(nn.Conv2d(ic, oc, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(oc),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.model(x)
    
    
class FastGANSimpleDecoder(nn.Module):
    def __init__(self, ic):
        super(FastGANSimpleDecoder, self).__init__()
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            FastGANUpBlk(ic, 128),
            FastGANUpBlk(128, 64),
            FastGANUpBlk(64, 64),
            FastGANUpBlk(64, 32),
            SpectralNorm(nn.Conv2d(32, 3, 3, 1, 1, bias=False)),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)
    

class FastGANSEBlk(nn.Module):
    def __init__(self, big_c, small_c, use_swish):
        super(FastGANSEBlk, self).__init__()
        self.branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            SpectralNorm(nn.Conv2d(small_c, big_c, 4, 1, 0, bias=False)),
            nn.SiLU() if use_swish else nn.LeakyReLU(0.1),
            SpectralNorm(nn.Conv2d(big_c, big_c, 1, 1, 0, bias=False)),
            nn.Sigmoid()
        )
    def forward(self, big, small):
        return big * self.branch(small)
    
    
class FastGANGenerator(nn.Module):
    def __init__(self, oc, nz, ngf, use_swish, sz):
        super(FastGANGenerator, self).__init__()
        self.sz = sz
        dims = {
            4: 16*ngf,
            8: 8*ngf,
            16: 4*ngf,
            32: 2*ngf,
            64: 2*ngf,
            128: ngf,
            256: ngf//2,
            512: ngf//4,
            1024: ngf//8
        }
        self.init_blk = FastGANGenInitBlk(nz, dims[4])
        self.blk_8 = FastGANUpBlk(dims[4], dims[8])
        self.blk_16 = FastGANUpBlk(dims[8], dims[16])
        self.blk_32 = FastGANUpBlk(dims[16], dims[32])
        self.blk_64 = FastGANUpBlk(dims[32], dims[64])
        self.blk_128 = FastGANUpBlk(dims[64], dims[128])
        self.blk_256 = FastGANUpBlk(dims[128], dims[256])
        self.blk_512 = FastGANUpBlk(dims[256], dims[512])
        self.blk_1024 = FastGANUpBlk(dims[512], dims[1024])
        
        self.se_128_8 = FastGANSEBlk(dims[128], dims[8], use_swish)
        self.se_256_16 = FastGANSEBlk(dims[256], dims[16], use_swish)
        self.se_512_32 = FastGANSEBlk(dims[512], dims[32], use_swish)
        
        self.last_conv1 = SpectralNorm(nn.Conv2d(dims[128], oc, 1, 1, 0, bias=False))
        self.last_conv2 = SpectralNorm(nn.Conv2d(dims[sz], oc, 3, 1, 1, bias=False))
    
    def forward(self, x, only_final=True):
        o = self.init_blk(x[..., None, None])
        o0 = self.blk_8(o)
        o1 = self.blk_16(o0)
        o2 = self.blk_32(o1)
        o = self.blk_64(o2)
        o = self.blk_128(o)
        
        o3 = self.se_128_8(o, o0)
        o = self.blk_256(o3)
        o = self.se_256_16(o, o1)
        
        # size is 512
        if self.sz >= 512:
            o = self.blk_512(o)
            o = self.se_512_32(o, o2)
        
        if self.sz >= 1024:
            o = self.blk_1024(o)
        
        out_128 = F.tanh(self.last_conv1(o3))
        out_final = F.tanh(self.last_conv2(o))
        
        if only_final:
            return out_final
        return out_128, out_final

    
class FastGANDiscriminator(nn.Module):
    def __init__(self, ic, ndf, use_swish, sz):
        super(FastGANDiscriminator, self).__init__()
        dims = {
            8: 16*ndf,
            16: 4*ndf,
            32: 2*ndf,
            64: ndf,
            128: ndf//2,
            256: ndf//4,
            512: ndf//8
        }
        self.init_blk = FastGANDiscInitBlk(ic, dims[512], dims[256], sz)
        self.blk_128 = FastGANDownBlk(dims[256], dims[128])
        self.blk_64 = FastGANDownBlk(dims[128], dims[64])
        self.blk_32 = FastGANDownBlk(dims[64], dims[32])
        self.blk_16 = FastGANDownBlk(dims[32], dims[16])
        self.blk_8 = FastGANDownBlk(dims[16], dims[8])
        
        self.se_32_256 = FastGANSEBlk(dims[32], dims[256], use_swish)
        self.se_16_128 = FastGANSEBlk(dims[16], dims[128], use_swish)
        self.se_8_64 = FastGANSEBlk(dims[8], dims[64], use_swish)
        
        self.last_conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(dims[8], dims[8], 1, 1, 0, bias=False)),
            nn.BatchNorm2d(dims[8]),
            nn.LeakyReLU(),
            SpectralNorm(nn.Conv2d(dims[8], 1, 4, 1, 0, bias=False))
        )
        
        small_dims = {
            8: 4*ndf,
            16: 2*ndf,
            32: ndf,
            64: ndf//2,
        }
        self.small_branch = nn.Sequential(
            SpectralNorm(nn.Conv2d(ic, small_dims[64], 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2),
            FastGANDownBlk(small_dims[64], small_dims[32]),
            FastGANDownBlk(small_dims[32], small_dims[16]),
            FastGANDownBlk(small_dims[16], small_dims[8])
        )
        self.last_conv2 = SpectralNorm(nn.Conv2d(small_dims[8], 1, 4, 1, 0, bias=False))
        
        self.dec1 = FastGANSimpleDecoder(dims[8])
        self.dec2 = FastGANSimpleDecoder(small_dims[8])
        self.dec3 = FastGANSimpleDecoder(dims[16])
    
    def forward(self, x, x_small, mode):
        o0 = self.init_blk(x)
        o1 = self.blk_128(o0)
        o2 = self.blk_64(o1)
        o = self.blk_32(o2)
        o = self.se_32_256(o, o0)
        o = self.blk_16(o)
        feat_16x16 = self.se_16_128(o, o1)
        o = self.blk_8(feat_16x16)
        feat_big = self.se_8_64(o, o2)
        out1 = self.last_conv1(feat_big)

        feat_small = self.small_branch(x_small)
        out2 = self.last_conv2(feat_small)
        out = torch.cat([out1, out2], 1)
        
        if mode == 'fake':
            return out
        elif mode == 'real':
            rec_big = self.dec1(feat_big)
            rec_small = self.dec2(feat_small)
            feat_8x8, corner = crop(feat_16x16)
            rec_crop = self.dec3(feat_8x8)
            return rec_big, rec_small, rec_crop, out, corner

def effective_bs(bs, num_gpus, num_nodes, dist_mode, grad_acc):
    if dist_mode == 'dp':
        eff_bs = bs
    elif dist_mode == 'ddp' or dist_mode == 'horovod':
        eff_bs = bs * num_gpus * num_nodes
    elif dist_mode == 'ddp2':
        eff_bs = bs * num_nodes
    
    eff_bs *= grad_acc
    return eff_bs

def inv_effective_bs(eff_bs, num_gpus, num_nodes, dist_mode, grad_acc):
    if dist_mode == 'dp':
        bs = eff_bs
    elif dist_mode == 'ddp' or dist_mode == 'horovod':
        bs = eff_bs // num_gpus // num_nodes
    elif dist_mode == 'ddp2':
        bs = eff_bs // num_nodes
    
    bs //= grad_acc
    return bs

def get_augmentation(aug_type, sz):
    if aug_type == 'mnist':
        tf = transforms.Compose([
            transforms.Resize((sz, sz)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    if aug_type == 'celeba':
        tf = transforms.Compose([
            transforms.Resize((sz, sz)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    return tf


class VisualizationCallback(Callback):
    def __init__(self, vis_interval):
        self.vis_interval = vis_interval

    def on_batch_end(self, trainer, pl_module):
        if trainer.global_step % self.vis_interval == 0:
            pl_module.visualize_step()

class FIDCallback(Callback):
    def __init__(self, fid_interval, fid_dir):
        self.fid_interval = fid_interval
        self.fid_dir = fid_dir

    def on_batch_end(self, trainer, pl_module):
        if trainer.global_step % self.fid_interval == 0:
            pl_module.fid_step(self.fid_dir)

class GANTrainer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        
        self.cfg = OmegaConf.load(args.config)
        
        if self.cfg.model.model_type == 'sngan_32':
            self.D = SNGANDiscriminator32(
                self.cfg.model.ic,
                self.cfg.model.d_dim
            ).apply(lambda m: weight_init(m, self.cfg.model.weight_init))

            self.G = SNGANGenerator32(
                self.cfg.model.ic,
                self.cfg.model.nz,
                self.cfg.model.g_dim,
                self.cfg.model.bottom
            ).apply(lambda m: weight_init(m, self.cfg.model.weight_init))
        
        elif self.cfg.model.model_type == 'dcgan_64':
            self.D = DCGANDiscriminator64(
            ).apply(lambda m: weight_init(m, self.cfg.model.weight_init))

            self.G = DCGANGenerator64(
                self.cfg.model.ic,
                self.cfg.model.nz
            ).apply(lambda m: weight_init(m, self.cfg.model.weight_init))
            
        elif self.cfg.model.model_type == 'fastgan':
            self.D = FastGANDiscriminator(
                self.cfg.model.ic,
                self.cfg.model.d_dim,
                self.cfg.model.use_swish,
                self.cfg.model.model_sz
            ).apply(lambda m: weight_init(m, self.cfg.model.weight_init))

            self.G = FastGANGenerator(
                self.cfg.model.ic,
                self.cfg.model.nz,
                self.cfg.model.g_dim,
                self.cfg.model.use_swish,
                self.cfg.model.model_sz
            ).apply(lambda m: weight_init(m, self.cfg.model.weight_init))
            
            self.vgg_loss = lpips.LPIPS(net='vgg')
        
        if self.cfg.training.diffaug is not None:
            self.diffaug = DiffAug(self.cfg.training.diffaug.split(','))
        else:
            self.diffaug = nn.Identity()
        
        for i, cur_data in enumerate([p.data for p in self.G.parameters()]):
            self.register_buffer(f'G_ema_{i}', cur_data.detach().clone())
        self.fixed_x = None

        # could be inconsistent with batchsize, also for fid calculation
        self.register_buffer('fixed_z', torch.randn(16, self.cfg.model.nz))
    
    def calculate_loss(self, mode, c_xr, c_xf):
        if self.cfg.training.loss_type == 'hinge':
            if mode == 'disc':
                return (F.relu(1 - c_xr) + F.relu(1 + c_xf)).mean()
            elif mode == 'gen':
                return -c_xf.mean()
        
        if self.cfg.training.loss_type == 'reg_hinge':
            if mode == 'disc':
                return (F.relu(torch.rand_like(c_xr) * 0.2 + 0.8 - c_xr) + 
                        F.relu(torch.rand_like(c_xf) * 0.2 + 0.8 + c_xf)).mean()
            elif mode == 'gen':
                return -c_xf.mean()
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        model_type = self.cfg.model.model_type
        if model_type == 'sngan_32' or model_type == 'dcgan_64':
            output = self.normal_step(batch, batch_idx, optimizer_idx)
        elif model_type == 'fastgan':
            output = self.fastgan_step(batch, batch_idx, optimizer_idx)
        return output
    
    def visualize_step(self):
        print('entered vis:', self.global_rank)
        xf = self.G(self.fixed_z)

        backup = copy.deepcopy([p.data for p in self.G.parameters()])
        for i, G_param in enumerate(self.G.parameters()):
            G_param.data.copy_(getattr(self, f'G_ema_{i}'))
        xf_ema = self.G(self.fixed_z)
        for G_param, backup_param in zip(self.G.parameters(), backup):
            G_param.data.copy_(backup_param)

        if self.global_rank == 0:
            grid_real = make_grid(self.fixed_x, 4, normalize=True)
            grid_fake = make_grid(xf, 4, normalize=True)
            grid_fake_ema = make_grid(xf_ema, 4, normalize=True)
            print('Image logged')
            trainer.logger.experiment.add_image('Real Images', grid_real, self.global_step)
            trainer.logger.experiment.add_image('Generated Images', grid_fake, self.global_step)
            trainer.logger.experiment.add_image('Generated Images EMA', grid_fake_ema, self.global_step)
        print('finished vis:', self.global_rank)
    
    def fid_step(self, fid_dir):
        print('entered fid:', self.global_rank)
        N = 5000
        bs = 16
        sizes = [bs] * (N//bs) + [N%bs]
        
        backup = copy.deepcopy([p.data for p in self.G.parameters()])
        for i, G_param in enumerate(self.G.parameters()):
            G_param.data.copy_(getattr(self, f'G_ema_{i}'))

        cnt = 0
        for sz in tqdm(sizes):
            z = torch.randn(sz, self.cfg.model.nz).to(self.device)
            xf_ema = self.G(z).detach().cpu().numpy()
            if self.global_rank == 0:
                for i in range(sz):
                    cur_image = (((xf_ema[i].transpose(1, 2, 0) + 1) / 2.0) * 255.0).astype('uint8')
                    cv2.imwrite(os.path.join(fid_dir, f'{cnt}.png'), cur_image)
                    cnt += 1
        
        for G_param, backup_param in zip(self.G.parameters(), backup):
            G_param.data.copy_(backup_param)
        
        if self.global_rank == 0:
            data_dir = self.cfg.data.dir
            paths = [os.path.join(data_dir, list(os.walk(data_dir))[0][1][0]), fid_dir]
            fid = fid_score.calculate_fid_given_paths(paths, 16, self.device, 2048)
            print('FID logged')
            self.logger.experiment.add_scalar(
                "FID", fid, self.global_step
            )
        print('finished fid:', self.global_rank)
    
    def normal_step(self, batch, batch_idx, optimizer_idx):
        xr, _ = batch
        z = torch.randn(xr.shape[0], self.cfg.model.nz).type_as(xr)
        
        if self.fixed_x is None:
            self.fixed_x = xr.clone()
        
        if optimizer_idx == 0:
            xr = self.diffaug(xr)
            xf = self.diffaug(self.G(z))
            c_xr = self.D(xr)
            c_xf = self.D(xf)
            loss = self.calculate_loss('disc', c_xr, c_xf)
            output = {"loss": loss}
            
            self.logger.experiment.add_scalar(
                "Discriminator loss",
                loss.item(),
                self.global_step)
        
        elif optimizer_idx == 1:
            xf = self.diffaug(self.G(z))
            c_xf = self.D(xf)
            loss = self.calculate_loss('gen', None, c_xf)
            output = {"loss": loss}
            
            self.logger.experiment.add_scalar(
                "Generator loss",
                loss.item(),
                self.global_step)
        
        return output
    
    def fastgan_step(self, batch, batch_idx, optimizer_idx):
        xr, _ = batch
        z = torch.randn(xr.shape[0], self.cfg.model.nz).type_as(xr)
        
        if self.fixed_x is None:
            self.fixed_x = xr.clone()
        
        if optimizer_idx == 0:
            xf_rec, xf = self.G(z, False)

            xr = self.diffaug(xr)
            xf = self.diffaug(xf)
            xf_rec = self.diffaug(xf_rec)
            xr_small = F.interpolate(xr, size=128, mode='nearest')

            rec_big, rec_small, rec_crop, c_xr, corner = self.D(xr, xr_small, 'real')
            c_xf = self.D(xf, xf_rec, 'fake')

            gan_loss = self.calculate_loss('disc', c_xr, c_xf)
            vgg_loss_1 = self.vgg_loss(rec_big, xr_small).sum()
            vgg_loss_2 = self.vgg_loss(rec_small, xr_small).sum()
            vgg_loss_3 = self.vgg_loss(rec_crop, F.interpolate(
                                           crop(xr_small, corner)[0],
                                           size=(rec_crop.shape[2:])
                                       )).sum()

            loss = gan_loss + vgg_loss_1 + vgg_loss_2 + vgg_loss_3
            
            output = {"loss": loss}
            
            self.logger.experiment.add_scalar(
                "Discriminator loss - GAN Loss",
                gan_loss.item(),
                self.global_step)
            
            self.logger.experiment.add_scalar(
                "Discriminator loss - VGG Loss",
                vgg_loss_1.item()+vgg_loss_2.item()+vgg_loss_3.item(),
                self.global_step)
            
            self.logger.experiment.add_scalar(
                "Discriminator loss",
                loss.item(),
                self.global_step)
        
        elif optimizer_idx == 1:
            xf_rec, xf = self.G(z, False)
            xf = self.diffaug(xf)
            xf_rec = self.diffaug(xf_rec)
            c_xf = self.D(xf, xf_rec, 'fake')
            loss = self.calculate_loss('gen', None, c_xf)
            output = {"loss": loss}
            
            self.logger.experiment.add_scalar(
                "Generator loss",
                loss.item(),
                self.global_step)
        
        return output
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        optimizer_idx = args[3]
        ema = self.cfg.training.ema
        if optimizer_idx == 1:
#             for G_param, G_ema_param in zip(self.G.parameters(), self.G_ema):
#                 print(G_ema_param.device)
#                 G_ema_param = G_ema_param * ema + G_param.data * (1-ema)
            for i, G_param in enumerate(self.G.parameters()):
                setattr(self, f'G_ema_{i}', getattr(self, f'G_ema_{i}') * ema + G_param.data * (1-ema))
        
    def configure_optimizers(self):
        lr = self.cfg.training.lr
        ttur_mult = self.cfg.training.ttur_mult
        beta1 = self.cfg.training.beta1
        beta2 = self.cfg.training.beta2
        weight_decay = self.cfg.training.weight_decay
        opt_type = self.cfg.training.opt_type
        n_critic = self.cfg.training.n_critic
        
        if opt_type == 'adam':
            opt_D = optim.Adam(self.D.parameters(), lr=lr*ttur_mult, betas=(beta1, beta2), weight_decay=weight_decay)
            opt_G = optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
        
        return {'optimizer': opt_D, 'frequency': n_critic}, {'optimizer': opt_G, 'frequency': 1}
    
    def train_dataloader(self):
        num_gpus = self.hparams.num_gpus
        num_nodes = self.hparams.num_nodes
        dist_mode = self.hparams.dist_mode
        grad_acc = self.hparams.grad_acc
        bs, n_workers = self.cfg.training.bs, self.cfg.training.n_workers
        bs = inv_effective_bs(bs, num_gpus, num_nodes, dist_mode, grad_acc)

        data_dir = self.cfg.data.dir
        data_type = self.cfg.data.data_type
        data_sz = self.cfg.data.sz
        tf = get_augmentation(data_type, data_sz)
        
        if data_type == 'mnist':
            ds = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
            dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=n_workers)
        elif data_type == 'celeba':
            ds = datasets.ImageFolder(data_dir, transform=tf)
            dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=n_workers)
        return dl


parser = ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config.yaml', help='.yaml config file')
parser.add_argument('-g', '--num-gpus', type=int, default=2, help='gpus')
parser.add_argument('-n', '--num-nodes', type=int, default=1, help='nodes')
parser.add_argument('-d', '--dist-mode', type=str, default='ddp', help='distributed modes')
parser.add_argument('-a', '--grad-acc', type=int, default=1, help='accumulated gradients')

parser.add_argument('-s', '--metric-image-path', type=str, help='path to save the generated images for evaluation')
parser.add_argument('-m', '--model-path-ckpt', type=str, help='model checkpoint path')
parser.add_argument('-r', '--resume-path-ckpt', type=str, default=None, help='resume training checkpoint path')
parser.add_argument('-e', '--experiment-name', type=str, default=None, help='experiment name')
parser.add_argument('-t', '--top-k-save', type=int, default=5, help='save top k')
parser.add_argument('-f', '--fast-dev-run', action='store_true', help='perform fast dev run')


args = parser.parse_args()
model = GANTrainer(args)
cfg = OmegaConf.load(args.config)

experiment_name = args.experiment_name
if experiment_name is None:
    experiment_name = datetime.now().strftime("%m%d%Y-%H:%M:%S")

ckpt_pth = os.path.join(args.model_path_ckpt, experiment_name)
fid_dir = os.path.join(args.metric_image_path, experiment_name)
log_dir = os.path.join(cfg.logging.log_dir, experiment_name)
os.makedirs(ckpt_pth, exist_ok=True)
os.makedirs(fid_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

lr_monitor = LearningRateMonitor('step')
ckpt_callback = ModelCheckpoint(
    filepath=ckpt_pth,
    save_top_k=-1,
    period=100
)
vis_callback = VisualizationCallback(cfg.logging.vis_interval)
fid_callback = FIDCallback(cfg.logging.fid_interval, fid_dir)

trainer = Trainer(
    logger=pl_loggers.TensorBoardLogger(log_dir, default_hp_metric=False),
    checkpoint_callback=ckpt_callback,
    weights_save_path=ckpt_pth,
    gpus=args.num_gpus,
    accumulate_grad_batches=args.grad_acc,
    distributed_backend=args.dist_mode,
    sync_batchnorm=True,
    resume_from_checkpoint=args.resume_path_ckpt,
    gradient_clip_val=cfg.training.gradient_clip,
    fast_dev_run=args.fast_dev_run,
    max_steps=cfg.training.iter_num * (1 + cfg.training.n_critic),
    callbacks=[lr_monitor],#, vis_callback, fid_callback],
    benchmark=True,
    num_sanity_val_steps=0,
    max_epochs=100000
)

trainer.fit(model)