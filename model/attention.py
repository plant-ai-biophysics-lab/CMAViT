from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from torch.nn.modules.utils import _pair
import tiktoken
import os 
from typing import Dict, Union
from torch.nn.functional import interpolate


import numpy as np
# from model import configs, engine
seed = 1987 
torch.manual_seed(seed)
np.random.seed(seed)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = DoubleConv(in_channels, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 128)
        self.conv5 = DoubleConv(128, 64)
        self.conv6 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(interpolate(x1, scale_factor=0.5, mode='bilinear', align_corners=True))
        x3 = self.conv3(interpolate(x2, scale_factor=0.5, mode='bilinear', align_corners=True))
        x4 = self.conv4(interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True))
        x5 = self.conv5(interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True))
        out = self.conv6(x5)
        return out


def exists(val):
    return val is not None

def uniq(arr):
    return {el: True for el in arr}.keys()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor





# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class RegressionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # reshape 
        self.norm =nn.LayerNorm(config.embed_dim*2)
        self.lfc = nn.Linear(self.config.embed_dim*2, 256, bias=True)  
        self.fcn = nn.GELU()
        self.fold = torch.nn.Fold(output_size=(self.config.img_size, self.config.img_size),
                                  kernel_size=1, dilation=1,
                                  padding=0, stride=1)

    def forward(self, x):
        x = self.norm(x)
        x = self.lfc(x)
        x = self.fcn(x)
        x = x.view(x.shape[0], 1, int(self.config.img_size**2))
        x = self.fold(x)

        return x

class MultiRegressionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.mask_modality != 'text': #
            n = 2
        else:
            n = 1

        self.regression = nn.ModuleDict(dict(
                norm =nn.LayerNorm(self.config.embed_dim*n), 
                lfc = nn.Linear(self.config.embed_dim*n, 256, bias=True),
                fcn = nn.GELU(),
                fold = torch.nn.Fold(output_size=(self.config.img_size, self.config.img_size),
                                  kernel_size=1, dilation=1,
                                  padding=0, stride=1)
            ))

    def forward(self, x_list):

        out = []
        for x in x_list:
            x = self.regression.norm(x)
            x = self.regression.lfc(x)
            x = self.regression.fcn(x)
            x = x.view(x.shape[0], 1, int(self.config.img_size**2))
            x = self.regression.fold(x)

            out.append(x)

        return out



def default(val, d):
    return val if val is not None else d


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if not glu:
            self.project_in = nn.Sequential(
                nn.Linear(dim, inner_dim),
                nn.GELU()
            )
        else:
            self.project_in = GEGLU(dim, inner_dim)  
        
        self.dropout = nn.Dropout(dropout)
        self.project_out = nn.Linear(inner_dim, dim_out)
        
        self.net = nn.Sequential(
            self.project_in,
            self.dropout,
            self.project_out
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        if isinstance(x, list):
            # Apply normalization and function fn to each tensor in the list
            return [self.fn(self.norm(xi)) for xi in x]
        else:
            # Single tensor processing as usual
            return self.fn(self.norm(x), **kwargs)

#========================================================================================#
#============================ Text Embedding and Encoder ================================#
#========================================================================================#

class TextEmbed(nn.Module):
    def __init__(self, 
                 config: Union[Dict], 
                 device = None):
        
        super().__init__()
        # Define device for the model and data
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.config = config
        
        self.TextEncoder = tiktoken.get_encoding('gpt2')
        
        self.proj = nn.Linear(1, config.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.dropout = nn.Dropout(config.proj_dropout)


    def forward(self, texts):
        encoded_texts = [self.TextEncoder.encode(text) for text in texts]
        max_length    = 249 #max(len(text) for text in encoded_texts)
        # padded_texts = [text + [0] * (max_length - len(text)) for text in encoded_texts]
        padded_texts = [text[:max_length] + [0] * (max_length - len(text)) for text in encoded_texts]
        texts_tensor = torch.tensor(padded_texts, dtype=torch.float32).to(self.device)
        texts_tensor = torch.unsqueeze(texts_tensor, dim=-1)
        
        B = texts_tensor.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        texts_tensor = self.proj(texts_tensor)
        texts_tensor = torch.cat((cls_tokens, texts_tensor), dim=1)

        embeddings = self.dropout(texts_tensor)


        return embeddings

class TextAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, text):
        b, n, _, h = *text.shape, self.heads
        qkv = self.to_qkv(text).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, attn

class TextEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, TextAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def forward(self, text):
        attn_scores = []
        for attn, ff in self.layers:
            out, attn_score = attn(text)
            attn_scores.append(attn_score)
            text = out + text
            text = ff(text) + text
        return self.norm(text), attn_scores

#========================================================================================#
#========================================================================================#
#========================================================================================#
class ImgEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, 
                 config):
        super().__init__()

        img_size = _pair(config.img_size)
        patch_size = _pair(config.patch_size)

        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        if config.multi_conv:
            self.proj = nn.Sequential(
                nn.Conv2d(config.in_channels, config.embed_dim // 4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(config.embed_dim // 4, config.embed_dim // 2, kernel_size=3, stride=2, padding = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(config.embed_dim // 2, config.embed_dim, kernel_size=3, stride=2, padding = 1),
            )
        else:
            self.proj = nn.Conv2d(config.in_channels, config.embed_dim, kernel_size=patch_size, stride=patch_size) #*temporal_obvs

        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches+1, config.embed_dim))
        self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.dropout             = nn.Dropout(config.proj_dropout)

    def forward(self, x):
        B = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)


        # x = rearrange(x, 'b c w h t -> b (c t) w h')
        x = self.proj(x).flatten(2).transpose(1, 2)

        x = torch.cat((cls_tokens, x), dim = 1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings

class STImgEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, 
                 config):
        super().__init__()

        img_size = _pair(config.img_size)
        patch_size = _pair(config.patch_size)
        temporal_obvs = 15
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * temporal_obvs
        
        self.proj = nn.Conv2d(config.in_channels * temporal_obvs, config.embed_dim, kernel_size=patch_size, stride=patch_size) 

        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches+1, config.embed_dim))
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
    
class AccImgEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, config, index):
        super().__init__()
        img_size = _pair(config.img_size)
        patch_size = _pair(config.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (index)
        
        self.num_patches = num_patches
        self.proj = nn.Conv2d(config.in_channels, config.embed_dim, kernel_size=patch_size, stride=patch_size) 
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches+1, config.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))   
        self.dropout = nn.Dropout(config.proj_dropout)

    def forward(self, x):
        B, T = x.shape[0], x.shape[-1]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Expand CLS tokens once
        # Rearrange to put time dimension next to batch
        x = rearrange(x, 'b c w h t -> (b t) c w h')
        # Apply convolution and adjust dimensions
        x = self.proj(x)  # Convolution output
        x = x.flatten(2).transpose(1, 2)  # Prepare for concatenation

        # Rearrange back into batches, handling non-contiguous layout with reshape
        x = rearrange(x, '(b t) n e -> b (t n) e', b=B)  # Reshape to original batch structure

        # Concatenate CLS tokens and add position embeddings
        embeddings = torch.cat((cls_tokens, x), dim=1) + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
    
class SpatialAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # if exists(met):
        #     met = self.to_qkv(met).chunk(3, dim=-1)
        #     qb, kb, _ = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=h), met)
        #     met = einsum('b h i d, b h j d -> b h i j', qb, kb) * self.scale
        #     dots += met
        
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class IncrementalSpatialAttention(nn.Module):
    def __init__(self, dim, heads=15, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        project_out = not (heads == 1 and dim_head == dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        # Removing the output projection to get individual head outputs
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        
        heads_outputs = []
        for head_idx in range(self.heads):
            # Each head processes an increasing window of tokens
            token_limit = 1 + ((head_idx + 1) * 4)  # Increasing window size for each head
            if token_limit > n:
                token_limit = n  # Ensure not to exceed number of tokens

            # Select tokens for this head
            qkv_head = [t[:, :token_limit] for t in qkv]
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_head)

            # Attention mechanism
            dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
            attn = dots.softmax(dim=-1)
            out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            # Append each head's output; we use nn.Identity() to skip any further projection
            out = self.to_out(out)

            
            heads_outputs.append(out)

        # Optionally, output could be rearranged or aggregated differently based on requirements
        return heads_outputs  # List of tensors, one per head

class IncrementalSpatialEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, IncrementalSpatialAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def average_resize(self, tensor, target_length):
        """Resize the tensor to the target length using mean pooling."""
        tensor = tensor.transpose(1, 2)  
        resized_tensor = F.adaptive_avg_pool1d(tensor, target_length)
        resized_tensor = resized_tensor.transpose(1, 2)
        return resized_tensor

    def forward(self, x):
        
        for attn, ff in self.layers:
            attn_out = attn(x)
            x_ts = [x[:, :1 + ((index + 1) * 4), :] + attn_out[index] for index in range(len(attn_out))]
            x_ts = [ff(out) + out for out in x_ts]
            x_ts_avg = [output if idx == 0 else self.average_resize(output, 4) for idx, output in enumerate(x_ts)]
            x = torch.cat(x_ts_avg, dim=1)

        return [self.norm(out) for out in x_ts]
    
class SpatialEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SpatialAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SpatialMetAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, met):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(met):
            met = self.to_qkv(met).chunk(3, dim=-1)
            qb, kb, _ = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=h), met)
            met = einsum('b h i d, b h j d -> b h i j', qb, kb) * self.scale
            dots += met
        
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, attn
    
class SpatialMetEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SpatialMetAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def forward(self, x, met = None):
        attn_scores = []
        for attn, ff in self.layers:
            out, attn_score = attn(x, met = met)
            attn_scores.append(attn_score)
            x = out + x
            x = ff(x) + x

        return self.norm(x), attn_scores
#========================================================================================#
#========================================================================================#
#========================================================================================#

class MetEmbed(nn.Module):
    """ Meteorological to Patch Embedding
    """
    def __init__(self, 
                 config,):
        super().__init__()


        self.num_patches = 15 

        self.proj = nn.Linear(4, config.embed_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches+1, config.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.dropout = nn.Dropout(config.proj_dropout)

    def forward(self, met):
        B = met.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        met = rearrange(met, 'b c t d -> b t (c d)')
        met = self.proj(met)#.flatten(2).transpose(1, 2)

        met = torch.cat((cls_tokens, met), dim = 1)
        embeddings = met + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings

class TempMetEmbed(nn.Module):
    """ 
    Meteorological to Patch Embedding.

    This module transforms meteorological data from a format [B, C, T, D] to a
    sequence of embeddings [B, 61, embed_dim], where:
    - B is the batch size
    - C is the number of channels (4)
    - T is the time dimension (15)
    - D is the depth of each feature (1)
    - 60 tokens are generated from the data, plus 1 class token.

    Parameters:
    - config: a configuration object with attributes embed_dim and proj_dropout.
    """
    def __init__(self, config):
        super().__init__()

        self.num_patches = 60  # Number of data-derived tokens
        self.proj = nn.Linear(4, config.embed_dim)  # Project C to embed_dim
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches + 1, config.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.dropout = nn.Dropout(config.proj_dropout)

    def forward(self, met):
        B = met.shape[0]
        # Expand class token to the batch size
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Rearrange and project the input
        met = rearrange(met, 'b c t d -> (b t) c d')
        met = self.proj(met)  # Shape: (B*T, embed_dim)

        # Reshape back to (B, T, embed_dim) and add the class token
        met = rearrange(met, '(b t) d -> b t d', b=B)
        met = torch.cat((cls_tokens, met), dim=1)  # Shape: (B, 61, embed_dim)

        # Add position embeddings and apply dropout
        embeddings = met + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings

class AccMetEmbed(nn.Module):
    """ Metadata to Patch Embedding """
    def __init__(self, config, index):
        super().__init__()
        num_patches = 4 * (index)
        self.proj = nn.Linear(1, config.embed_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.dropout = nn.Dropout(config.proj_dropout)

    def forward(self, met):
        B = met.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Rearrange and process
        met = rearrange(met, 'b c t d -> b t (c d)')
        met = torch.unsqueeze(met, dim=3)  # Add a new dimension for linear projection
        met = self.proj(met)
        met = rearrange(met, 'b t n e -> b (t n) e')  # Flatten time and feature dimensions

        # Concatenate CLS tokens and position embeddings
        embeddings = torch.cat((cls_tokens, met), dim=1) + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings
    
    
class MetAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
    
class MetEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MetAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class IncrementalMetAttention(nn.Module):
    def __init__(self, dim, heads=15, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        project_out = not (heads == 1 and dim_head == dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        # Removing the output projection to get individual head outputs
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        
        heads_outputs = []
        for head_idx in range(self.heads):
            # Each head processes an increasing window of tokens
            token_limit = 1 + ((head_idx + 1) * 4)  # Increasing window size for each head
            if token_limit > n:
                token_limit = n  # Ensure not to exceed number of tokens

            # Select tokens for this head
            qkv_head = [t[:, :token_limit] for t in qkv]
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_head)

            # Attention mechanism
            dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
            attn = dots.softmax(dim=-1)
            out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            # Append each head's output; we use nn.Identity() to skip any further projection
            out = self.to_out(out)

            
            heads_outputs.append(out)

        # Optionally, output could be rearranged or aggregated differently based on requirements
        return heads_outputs  # List of tensors, one per head

class IncrementalMetEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, IncrementalMetAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def average_resize(self, tensor, target_length):
        """Resize the tensor to the target length using mean pooling."""
        tensor = tensor.transpose(1, 2)  
        resized_tensor = F.adaptive_avg_pool1d(tensor, target_length)
        resized_tensor = resized_tensor.transpose(1, 2)
        return resized_tensor

    def forward(self, x):
        
        for attn, ff in self.layers:
            attn_out = attn(x)
            x_ts = [x[:, :1 + ((index + 1) * 4), :] + attn_out[index] for index in range(len(attn_out))]
            x_ts = [ff(out) + out for out in x_ts]
            x_ts_avg = [output if idx == 0 else self.average_resize(output, 4) for idx, output in enumerate(x_ts)]
            x = torch.cat(x_ts_avg, dim=1)

        return [self.norm(out) for out in x_ts]
    
#========================================================================================#
#========================================================================================#
#========================================================================================#
class MultiModalAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        # Also return the attention scores, reshaped back to [B, N, H, T] where N is query length and T is context length
        # attn = rearrange(attn, '(b h) n j -> b h n j', b=x.size(0), h=h)

        return self.to_out(out), attn

class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, met = None):
        b, n, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(met):
            met = self.to_qkv(met).chunk(3, dim=-1)
            qb, kb, _ = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=h), met)
            met = einsum('b h i d, b h j d -> b h i j', qb, kb) * self.scale
            dots += met

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class MultiModalTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, context_dim=9, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiModalAttention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head,
                                                 dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def forward(self, x, context=None):
        attn_scores = []
        for attn, ff in self.layers:
            out, attn_score = attn(x, context=context)
            attn_scores.append(attn_score)
            x = out + x
            x = ff(x) + x
        return self.norm(x), attn_scores

class TemporalTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, TemporalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def forward(self, x, met=None):
        for attn, ff in self.layers:
            x = attn(x, met=met) + x
            x = ff(x) + x
        return self.norm(x)