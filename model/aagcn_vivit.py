import numpy as np
import copy as cp

import torch
from torch import nn
from torch.nn.modules.utils import _triple

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from model.layers.module import Attention, PreNorm, FeedForward
from model.builder import BACKBONES
from utils.graph import Graph
from model.layers.aagcn import AAGCN
from utils.misc import cache_checkpoint
from mmcv.runner import load_checkpoint

EPS = 1e-4


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def exists(val):
    return val is not None


@BACKBONES.register_module()
class AAGCN_vivit(nn.Module):
    def __init__(
        self,
        graph_cfg,
        num_classes,
        dim,
        spatial_depth,
        temporal_depth,
        heads,
        mlp_dim,
        pool = 'cls',
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        pretrained=None,
    ):
        super().__init__()

        # graph_cfg=dict(layout='nturgb+d', mode='spatial')
        self.aagcn_backbone = AAGCN(graph_cfg=graph_cfg)

        # patch_dim = channels * patch_height * patch_width * frame_patch_size
        

        self.global_average_pool = pool == 'mean'

        self.rearrange = Rearrange('b m c t v -> b t v (m c)')

        num_person = 2
        patch_dim = 256 * num_person
        self.linear = nn.Linear(patch_dim, dim) # dim usually 1024

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_frame_patches, num_image_patches, dim))
        
 
        self.dropout = nn.Dropout(emb_dropout)

        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None

        self.spatial_transformer = Transformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout)
        self.temporal_transformer = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )
        self.pretrained = pretrained

    def init_weights(self):
        # TODO
        # pass
        if isinstance(self.pretrained, str):
            # pass
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x):

        x = self.aagcn_backbone(x)

        # transformer
        x = self.rearrange(x)

        x = self.linear(x)

        b, f, n, _ = x.shape

        # x = x + self.pos_embedding

        if exists(self.spatial_cls_token):
            spatial_cls_tokens = repeat(self.spatial_cls_token, '1 1 d -> b f 1 d', b = b, f = f)
            x = torch.cat((spatial_cls_tokens, x), dim = 2)

        # x += self.pos_embedding[:, :, :(n + 1)]

        x = self.dropout(x)

        x = rearrange(x, 'b f n d -> (b f) n d')

        # attend across space

        x = self.spatial_transformer(x)

        x = rearrange(x, '(b f) n d -> b f n d', b = b)

        # excise out the spatial cls tokens or average pool for temporal attention

        x = x[:, :, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b f d', 'mean')

        # append temporal CLS tokens

        if exists(self.temporal_cls_token):
            temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b = b)

            x = torch.cat((temporal_cls_tokens, x), dim = 1)

        # attend across time

        x = self.temporal_transformer(x)

        # excise out temporal cls token or average pool

        x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean')

        x = self.to_latent(x)

        # x = self.mlp_head(x)

        return x # 
    

