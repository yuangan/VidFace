from .token_transformer import Token_transformer, Detoken_transformer
from .token_performer import Token_performer
from .transformer_block import Block, get_sinusoid_encoding
import numpy as np
from timm.models.layers import trunc_normal_
import torch
import torch.nn as nn

class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if tokens_type == 'alltrans':
            print('adopt alltrans encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

            self.project0 = nn.Linear(in_chans, 16)
            
            self.attention1 = Token_transformer(dim=16*3*3, in_dim=token_dim//8, num_heads=8, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim*3*3, in_dim=token_dim//8, num_heads=8, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        self.num_patches = (img_size // (1)) * ((img_size//4*3) // (1))  # there are 3 sfot split, stride are 4,2,2 seperately
        self.h = img_size
        self.w = img_size // 4 * 3

    def forward(self, x):
        b, t, c, _, _ = x.size()
        x = x.reshape(b*t, c, self.h*self.w).transpose(1,2)
        x = self.project0(x)
        # step0: soft split
        x = x.transpose(1, 2).reshape(b*t, 16, self.h, self.w) 
        
        x1 = self.soft_split0(x).transpose(1, 2) # b, dim, sequence -> b, sequence, dim
        _, seq, dim1 = x1.shape
        x1 = x1.reshape(b, t*seq, dim1)
        # iteration1: re-structurization/reconstruction
        x1 = self.attention1(x1)
        _, seq, dim2 = x1.shape
        x2 = x1.reshape(b*t, seq//t, dim2)
        x2 = x2.transpose(1, 2).reshape(b*t, dim2, self.h, self.w)
        
        # iteration1: soft split
        x2 = self.soft_split1(x2).transpose(1, 2)
        _, seq, dim2 = x2.shape
        x2 = x2.reshape(b, t*seq, dim2)
        # iteration2: re-structurization/reconstruction
        x2 = self.attention2(x2)
        _, seq, dim3 = x2.shape
        
        x3 = x2.reshape(b*t, seq//t, dim3)
        x3 = x3.transpose(1, 2).reshape(b*t, dim3, self.h, self.w)
        
        # iteration2: soft split
        x3 = self.soft_split2(x3).transpose(1, 2) # b*t, sequence, dim

        # final tokens
        x3 = self.project(x3)
        return x1, x2, x3

class DeU_module(nn.Module):
    """
        DeU decoding module is composed of two key modules:
            - Detoken Transformer (DeT)
            - Tokens to Image (T2I).
    """
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        self.h = img_size
        self.w = img_size // 4 * 3

        if tokens_type == 'alltrans':
            print('adopt alltrans decoder for tokens-to-token')
            self.unproject1 = nn.Linear(embed_dim+10, token_dim * 3 * 3+10) # the channel could be lower
            self.detoken_trans = Detoken_transformer(dim=1088, in_dim=8192//8, num_heads=8, mlp_ratio=1.0)
            self.norm = nn.LayerNorm(8192)
            self.token_to_image = nn.Fold(output_size=(self.h*8, self.w*8), kernel_size=(16, 16), stride=(8, 8), padding=(4, 4))
        else:
            raise NotImplementedError(
                'other transformer is not supported now.')

        self.m1 = None
        self.m2 = None
        self.m3 = None

    def forward(self, x1, x2, x3, x):
        upx3 = x
        upx3 = self.unproject1(upx3)
        upx3 = torch.cat((upx3, x1, x2, x3), dim=2)
        upx3 = self.detoken_trans(upx3)
        upx3 = self.norm(upx3)
        t = 7
        b, tl, c = upx3.shape
        upx3 = upx3.reshape(b*t, tl//t, c).transpose(1, 2) # b, sequence, dim -> b, dim, sequence
        
        upx3 = self.token_to_image(upx3) # b*t c=68 2*h 2*w
        return upx3



class VIDFACE(nn.Module):
    def __init__(self, img_size=16, tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, token_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.tokens_to_token = T2T_module(
                img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim)
        self.deu = DeU_module(
                img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim)
        num_patches = self.tokens_to_token.num_patches

        # learnable position embedding
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # refiner block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim+10, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)
        self.device = torch.cuda.current_device()

        full = torch.arange(0, 192).to(self.device)
        self.full_h = (full//12).float()/16. - 0.5
        self.full_w = (full%12).float()/12. - 0.5

        self.down1 = nn.Linear(embed_dim+10, embed_dim)
        self.down2 = nn.Linear(embed_dim, 32)
        self.down3 = nn.Linear(32, 16)
        self.lmhead = nn.Linear(192*16, 10)
        self.num_iter = 2
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.03)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def gen_lmd(self, landms):
        # generate landmark position embedding
        distance = []
        for i in range(5):
            pw = (self.full_w - landms[:, 2*i].unsqueeze(1))
            ph = (self.full_h - landms[:, 2*i+1].unsqueeze(1))
            distance.append(pw)
            distance.append(ph)
        distance = torch.stack(distance, dim=0)
        distance = distance.permute(1,2,0)
        return distance
    
    def forward_features(self, x):
        B = x.shape[0]
        # Augmented T2T transformer
        x1, x2, x3 = self.tokens_to_token(x)

        x = x3 + self.pos_embed
        x = self.pos_drop(x)
        bt, l, d = x.shape
        x = x.reshape(bt//7, 7*l, d)
        
        # Refiner
        lmks = []
        lmk = torch.zeros(bt, 10).to(self.device)
        for i in range(self.num_iter):
            # RLPE: recurrent landmark position embedding
            lmd = self.gen_lmd(lmk)
            lmd = lmd.reshape(bt//7, 7*l, 10)
            x = torch.cat((x, lmd), dim=2)
            for blk in self.blocks:
                x = blk(x)
            x = self.down1(x)
            x_down = self.down2(x)
            x_down = self.down3(x_down)
            x_down = x_down.reshape(bt, l*16)
            lmk = self.lmhead(x_down)
            lmks.append(lmk)
        x = self.norm(x)
        
        # Decoder
        lmd = lmd.reshape(bt//7, 7*l, 10)
        x = torch.cat((x, lmd), dim=2)
        x3 = x3.reshape(bt//7, 7*l, d)
        x = self.deu(x1, x2, x3, x)
        
        return x, lmks

    def forward(self, x):
        x, lmks = self.forward_features(x)
        return x, lmks
