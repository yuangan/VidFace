import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F

from basicsr.models.archs.arch_util import trunc_normal_
from basicsr.models.archs.unt2t_vit_h48t_norm_l10 import VIDFACE
from timm.models.layers import trunc_normal_

class VidFace(nn.Module):
    """VidFace network structure for video super-resolution.

    Now only support X8 upsampling factor.
    Paper:
        VidFace: A Full-Transformer Solver for Video FaceHallucination with Unaligned Tiny Snapshots

    Args:
    """

    def __init__(self,
                 dropout=0.0):
        super(VidFace, self).__init__()

        torch.autograd.set_detect_anomaly(True)
        self.dropout = dropout

        # extrat qk/c features
        self.vidface = VIDFACE(tokens_type='alltrans', embed_dim=374, depth=4, num_heads=8, mlp_ratio=3.)
        self.linear_hr = nn.Linear(32, 64)
        self.linear_last = nn.Linear(64, 3)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # gpu device
        self.device = torch.cuda.current_device()

    def unflatten(self, x, h, w):
        b, _, c = x.shape
        return x.permute(0, 2, 1).reshape(b, c, h, w)

    def flatten(self, x):
        b, c, h, w = x.shape
        return x.reshape(b, c, h*w).permute(0, 2, 1)

    def forward(self, x):
        b, t, c, h, w = x.size()
        
        x_ori = x.view(b*t, c, h, w).contiguous()
        x, lmks = self.vidface(x)
        bt, dim, h, w = x.shape
        
        out = self.lrelu(self.linear_hr(self.flatten(x)))
        out = self.linear_last(out)
        out = self.unflatten(out, h, w)
        base = F.interpolate(
                x_ori, scale_factor=8, mode='bilinear', align_corners=False)
        out += base

        out = out.view(b, t, c, h, w)
        return out, lmks, None
# 31 now
