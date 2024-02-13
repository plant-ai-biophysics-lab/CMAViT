import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
import pdb
import math
import numpy as np
from transformers import BertModel, BertTokenizer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", cache_dir = '/data2/hkaman/DiT/')
# tokenizer = AutoTokenizer.from_pretrained("/data2/hkaman/DiT/models--bert-base-cased/")

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# seed = 1987
# torch.manual_seed(seed) # important 
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}

def default_conv(in_channels, out_channels, kernel_size, bias=True, groups=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding = (kernel_size // 2), bias=bias, groups=groups)

class img_unfold(nn.Module):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    def __init__(self, config):
        super().__init__()
        self.kernel_size = config.kernel_size
        self.dilation    = config.dilation
        self.stride      = config.stride

        self.unfold = torch.nn.Unfold(kernel_size = self.kernel_size,
                            dilation    = self.dilation,
                            padding     = 0,
                            stride      = self.stride)
        

    def same_padding(self, x, ksizes = [3, 3], strides = [1, 1], rates = [1, 1]):
        assert len(x.size()) == 4
        batch_size, channel, rows, cols = x.size()
        out_rows = (rows + strides[0] - 1) // strides[0]
        out_cols = (cols + strides[1] - 1) // strides[1]
        effective_k_row = (ksizes[0] - 1) * rates[0] + 1
        effective_k_col = (ksizes[1] - 1) * rates[1] + 1
        padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
        padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
        # Pad the input
        padding_top = int(padding_rows / 2.)
        padding_left = int(padding_cols / 2.)
        padding_bottom = padding_rows - padding_top
        padding_right = padding_cols - padding_left
        paddings = (padding_left, padding_right, padding_top, padding_bottom)
        x = torch.nn.ZeroPad2d(paddings)(x)
        return x
    
    def forward(self, x):
        x = self.same_padding(x)
        x = self.unfold(x)
        return x

class img_fold(nn.Module):

    def __init__(self, config):
        super().__init__()
        out_size    = (180, 360) 
        kernel_size = 1 #config.kernel_size
        dilation    = config.dilation
        stride      = config.stride

        self.fold = torch.nn.Fold(output_size=out_size,
                           kernel_size= kernel_size,
                           dilation   = dilation,
                           padding    = 0,
                           stride     = stride)

    def forward(self, x):

        x = self.fold(x)

        return x

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(config.Proj_drop),
            nn.Linear(512, config.embed_dim),
            nn.Dropout(config.Proj_drop)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()

        dim_head = config.embed_dim // config.num_heads
        inner_dim = dim_head *  config.num_heads
        project_out = not (config.num_heads == 1 and dim_head == config.embed_dim)

        self.heads = config.num_heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(config.embed_dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(config.Attn_drop)

        self.to_qkv = nn.Linear(config.embed_dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, config.embed_dim),
            nn.Dropout(config.Attn_drop)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.embed_dim)
        self.layers = nn.ModuleList([])
        for _ in range(config.num_layers):
            self.layers.append(nn.ModuleList([
                Attention(config),
                FeedForward(config)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class sp_patch_embedding(nn.Module):
    def __init__(self, config):
        super(sp_patch_embedding, self).__init__()

        image_height, image_width = pair(16)
        patch_height, patch_width = pair(8)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        temporal_res = 15
        num_patches = (image_height // patch_height) * (image_width // patch_width) * temporal_res + 4
        patch_dim = config.in_channels * patch_height * patch_width
        assert config.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, config.embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim))
        self.dropout = nn.Dropout(config.Proj_drop)
        self.text_embedings = nn.Embedding(128, config.embed_dim)

    def forward(self, x, text_data):
        B = x.shape[0]
        T = x.shape[-1]

        x_stack = None
        for t in range(T): 
            this_time = x[:, :, :, :, t]
            this_time = self.to_patch_embedding(this_time)

            if x_stack is None: 
                x_stack = this_time
            else:
                x_stack = torch.cat((x_stack, this_time), dim = 1)
        b, n, _ = x_stack.shape

        # encoded_text_input = tokenizer(text_data, padding=True, truncation=True, return_tensors="pt")
        aux = self.text_embedings(text_data)

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = B)
 
        x = torch.cat((cls_tokens, x_stack, aux), dim=1)
        x += self.pos_embedding#[:, :(n + 1)]
        x = self.dropout(x)

        # x = torch.cat((x, aux), dim=1)

        return x


class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.patch_embedding = sp_patch_embedding(config)
        self.transformer = Transformer(config)

        self.pool = config.pool
        self.to_latent = nn.Identity()
        self.regression_head = nn.Linear(config.embed_dim, 256)

    def forward(self, x, text_data):

        x = self.patch_embedding(x, text_data)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        x = self.regression_head(x)
        x = x.view(x.shape[0], 1, 16, 16)


        return x# self.mlp_head(x)