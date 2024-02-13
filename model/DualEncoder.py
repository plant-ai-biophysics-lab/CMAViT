import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import pdb
import math
import numpy as np
from transformers import BertModel, BertTokenizer
import torch.nn.init as init
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Mlp
import torch.distributed as dist




def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

seed = 1987 + get_rank()
torch.manual_seed(seed)
np.random.seed(seed)


class ImgEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, 
                 config,):
        super().__init__()

        img_size = _pair(config.img_size)
        patch_size = _pair(config.patch_size)
        temporal_obvs = 15
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) 
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        if config.multi_conv:
            self.proj = nn.Sequential(
                nn.Conv2d(config.in_channels*temporal_obvs, config.embed_dim // 4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(config.embed_dim // 4, config.embed_dim // 2, kernel_size=3, stride=2, padding = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(config.embed_dim // 2, config.embed_dim, kernel_size=3, stride=2, padding = 1),
            )
        else:
            self.proj = nn.Conv2d(config.in_channels*temporal_obvs, config.embed_dim, kernel_size=patch_size, stride=patch_size)

        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches+1, config.embed_dim))
        self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.dropout             = nn.Dropout(config.proj_dropout)

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)


        x = rearrange(x, 'b c w h t -> b (c t) w h')
        x = self.proj(x).flatten(2).transpose(1, 2)

        x = torch.cat((cls_tokens, x), dim = 1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings

class TextEmbed(nn.Module):

    def __init__(self, config):
        super().__init__()

        n_patches  = 4 

        assert n_patches is not None, ('Number of Patches Can NOT be None!')

        self.text_embedings = nn.Embedding(64, config.embed_dim)

        self.text_position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.embed_dim))
        self.text_cls_token           = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.text_dropout             = nn.Dropout(0.1)

    def forward(self, text):
        B = text.shape[0] 
        text_cls_tokens = self.text_cls_token.expand(B, -1, -1)
        aux = self.text_embedings(text) 
        aux = torch.cat((text_cls_tokens, aux), dim = 1)
        text_embeddings = aux + self.text_position_embeddings
        text_embeddings = self.text_dropout(text_embeddings)
        return text_embeddings

class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        self.c_attn = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=True)

        self.c_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.proj_dropout)
        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim
        self.dropout = config.proj_dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
    #     self._init_weights()

    # def _init_weights(self):
    #     # Use kaiming_normal_ for weights initialization
    #     nn.init.kaiming_normal_(self.c_attn.weight, mode='fan_in', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.c_proj.weight, mode='fan_in', nonlinearity='relu')
        
    #     # Initialize biases to zero
    #     nn.init.zeros_(self.c_attn.bias)
    #     nn.init.zeros_(self.c_proj.bias)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        q, k, v  = self.c_attn(x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class CrossAttention(nn.Module):
    def __init__(
        self,
        config,
        parallel_ff=False,

    ):
        super().__init__()
        self.heads = config.num_heads
        dim_head = int(config.embed_dim / config.num_heads)
        self.scale = dim_head ** -0.5
        inner_dim = config.num_heads * dim_head
        text_dim = default(config.embed_dim, config.embed_dim)

        self.attn_dropout = nn.Dropout(config.Attn_drop)
        self.resid_dropout = nn.Dropout(config.Proj_drop)


        self.norm = nn.LayerNorm(config.embed_dim)
        self.text_norm = nn.LayerNorm(text_dim) #if norm_text else nn.Identity()

        self.to_q = nn.Linear(config.embed_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(text_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, config.embed_dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = 4 * config.embed_dim

        self.ff = nn.Sequential(
            nn.Linear(config.embed_dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, config.embed_dim, bias=False)
        ) if parallel_ff else None

    def forward(self, x, text):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # pre-layernorm, for queries and context
        x = self.norm(x)
        text = self.text_norm(text)
        # get queries
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        # scale
        q = q * self.scale

        # get key / values
        k, v = self.to_kv(text).chunk(2, dim=-1)
        # query / key similarity
        sim = torch.einsum('b h i d, b j d -> b h i j', q, k)

        # attention
        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        # aggregate
        out = torch.einsum('b h i j, b j d -> b h i d', attn, v)
        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # add parallel feedforward (for multimodal layers)
        if exists(self.ff):
            out = out + self.ff(x)

        out = self.resid_dropout(out)
        return out

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.attn = SelfAttention(config)
        self.drop_path = DropPath(config.drop_path) if config.drop_path > 0. else nn.Identity()
        self.ln_2 = nn.LayerNorm(config.embed_dim)
        self.mlp = Mlp(in_features = config.embed_dim, 
                       hidden_features = config.embed_dim*4, 
                       act_layer = nn.GELU, 
                       drop = config.proj_dropout)


    def forward(self, x):
        x = x + self.drop_path(self.attn(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x

class CrossAttnBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.attn = CrossAttention(config)
        self.drop_path = DropPath(config.drop_path) if config.drop_path > 0. else nn.Identity()
        self.ln_2 = nn.LayerNorm(config.embed_dim)
        self.mlp = Mlp(in_features = config.embed_dim, 
                       hidden_features = config.embed_dim*4, 
                       act_layer = nn.GELU, 
                       drop = config.Proj_drop)


    def forward(self, x, text):
        x = x + self.drop_path(self.attn(x, text))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x
    
class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # reshape 
        self.lfc = nn.Linear(self.config.embed_dim, 256, bias=True)  
        self.fcn = nn.GELU()
        self.fold = torch.nn.Fold(output_size=(self.config.img_size, self.config.img_size),
                                  kernel_size=1, dilation=1,
                                  padding=0, stride=1)


    def forward(self, x):
        x = self.lfc(x)
        x = self.fcn(x)
        x = x.view(x.shape[0], 1, int(self.config.img_size**2))
        x = self.fold(x)

        return x
    
class ViT(nn.Module):
    def __init__(self, config):
        super(ViT, self).__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            img_embed = ImgEmbed(self.config), 
            h = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            ln_f = nn.LayerNorm(config.embed_dim),
            head = RegressionHead(config)
        ))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.transformer.img_embed(x) 
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x = x.mean(dim = 1) 
        
        x = self.transformer.head(x)

        return x

class CrossAttnViT(nn.Module):
    def __init__(self, config):
        super(CrossAttnViT, self).__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            text_embed = TextEmbed(self.config),
            img_embed = ImgEmbed(self.config), 
            img = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            txt = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            # h = nn.ModuleList([CrossAttnBlock(config) for _ in range(config.num_layers)]),
            ln_f = nn.LayerNorm(config.embed_dim),
            head = RegressionHead(config)
        ))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, text):
        x = self.transformer.img_embed(x) 
        text = self.transformer.text_embed(text) 

        for img_block in self.transformer.img:
            x = img_block(x)

        for txt_block in self.transformer.txt:
            text = txt_block(text)

        crs = torch.cat((x, text), dim =1)
        crs = self.transformer.ln_f(crs)
        crs = crs.mean(dim = 1) 
        
        crs = self.transformer.head(crs)

        return crs