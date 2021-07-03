import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F

from basicsr.models.archs.arch_util import (ResidualBlockNoBN,
                                            make_layer,
                                            LargeEncoder, DownSampler2, trunc_normal_)
from basicsr.models.archs.unt2t_vit_h48t_norm_l10 import T2T_ViT2
from basicsr.models.archs.corr import CorrBlock, HalfCorrBlock
from timm.models.layers import trunc_normal_

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class warp(torch.nn.Module):
    def __init__(self, h, w, cuda_flag):
        super(warp, self).__init__()
        self.height = h
        self.width = w
        if cuda_flag:
            self.addterm = self.init_addterm().cuda()
        else:
            self.addterm = self.init_addterm()

    def init_addterm(self):
        n = torch.FloatTensor(list(range(self.width)))
        horizontal_term = n.expand((1, 1, self.height, self.width))  # 第一个1是batch size
        n = torch.FloatTensor(list(range(self.height)))
        vertical_term = n.expand((1, 1, self.width, self.height)).permute(0, 1, 3, 2)
        addterm = torch.cat((horizontal_term, vertical_term), dim=1)
        return addterm

    def forward(self, frame, flow):
        """
        :param frame: frame.shape (batch_size=1, n_channels=3, width=256, height=448)
        :param flow: flow.shape (batch_size=1, n_channels=2, width=256, height=448)
        :return: reference_frame: warped frame
        """
        if True:
            flow = flow + self.addterm
        else:
            self.addterm = self.init_addterm()
            flow = flow + self.addterm

        horizontal_flow = flow[0, 0, :, :].expand(1, 1, self.height, self.width)  # 第一个0是batch size
        vertical_flow = flow[0, 1, :, :].expand(1, 1, self.height, self.width)

        horizontal_flow = horizontal_flow * 2 / (self.width - 1) - 1
        vertical_flow = vertical_flow * 2 / (self.height - 1) - 1
        flow = torch.cat((horizontal_flow, vertical_flow), dim=1)
        flow = flow.permute(0, 2, 3, 1)
        reference_frame = torch.nn.functional.grid_sample(frame, flow)
        return reference_frame

size_1 = 128
size_2 = 256
size_3 = 512

class EDVR(nn.Module):
    """EDVR network structure for video super-resolution.

    Now only support X4 upsampling factor.
    Paper:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_out_ch (int): Channel number of output image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_frame (int): Number of input frames. Default: 5.
        deformable_groups (int): Deformable groups. Defaults: 8.
        num_extract_block (int): Number of blocks for feature extraction.
            Default: 5.
        num_reconstruct_block (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: 2.
        hr_in (bool): Whether the input has high resolution. Default: False.
        with_predeblur (bool): Whether has predeblur module.
            Default: False.
        with_tsa (bool): Whether has TSA module. Default: True.
    """

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 deformable_groups=8,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 center_frame_idx=2,
                 hr_in=False,
                 with_predeblur=False,
                 with_tsa=True,
                 dropout=0.0):
        super(EDVR, self).__init__()

        torch.autograd.set_detect_anomaly(True)
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        self.hr_in = hr_in
        self.dropout = dropout

        # RAFT Parameter
        self.corr_radius = 4
        self.num_feat = num_feat
        
        # RAFT Network
        # extrat qk/c features
        self.feature_extractor_qk = T2T_ViT2(tokens_type='alltrans', embed_dim=374, depth=4, num_heads=8, mlp_ratio=3.)
        
        # self.linear1 = nn.Linear(128, num_feat*4)
        # self.linear2 = nn.Linear(num_feat, num_feat*4)
        # self.linear3 = nn.Linear(num_feat, 64*4)
        # self.pixel_shuffle = nn.PixelShuffle(2)

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
        x, lmks = self.feature_extractor_qk(x)
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
