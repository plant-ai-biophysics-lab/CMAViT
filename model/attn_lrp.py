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
from typing import Union, Dict, List
# torch.autograd.set_detect_anomaly(True)

import numpy as np
from model import configs, engine
from model.layers_ours import *




seed = 1987 + engine.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"




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

def default(x, y):
    return x if x is not None else y

def exists(val):
    return val is not None




class MultiRegressionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.mask_modality != 'text':
            n = 2
        else:
            n = 1

        self.regression = nn.ModuleDict(dict(
                norm = LayerNorm(self.config.embed_dim), 
                lfc  = Linear(self.config.embed_dim, 256, bias=True),
                fcn  = GELU(),
                # fold = torch.nn.Fold(output_size=(self.config.img_size, self.config.img_size),
                #                   kernel_size=1, dilation=1,
                #                   padding=0, stride=1)
            ))

    def forward(self, x_list):

        out = []
        for x in x_list:
            # if not x.requires_grad:
            #     x.requires_grad_(True)  # Ensure gradients are tracked if not already
            x = self.regression.norm(x)
            x = self.regression.lfc(x)
            x = self.regression.fcn(x)
            x = x.view(x.shape[0], 1, int(self.config.img_size), int(self.config.img_size))
            # x = self.regression.fold(x)

            out.append(x)

        return out
    
    def relprop(self, cam_list):
        relprops = []
        for cam in cam_list:
            # Reverse Fold operation
            cam = cam.view(cam.shape[0], -1, 1)  # Adjust shape to match the view before fold in forward
            # Simulate the reverse of Fold as an unfold operation or use stored outputs to map back
            # cam = F.unfold(cam, kernel_size=1, dilation=1, padding=0, stride=1)

            # Reverse GELU activation: Consider using an approximation or stored activations for exact gradients
            cam = self.regression.fcn(cam, reverse=True)

            # Reverse Linear transformation
            cam = self.regression.lfc.relprop(cam)

            # Reverse Layer Normalization
            cam = self.regression.norm.relprop(cam)

            relprops.append(cam)

        return relprops

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = GELU()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam

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

    def relprop(self, cam, **kwargs):
        if isinstance(cam, list):
            # If cam is a list, propagate relevance through each element
            return [self.fn.relprop(self.norm.relprop(cam_i, **kwargs), **kwargs) for cam_i in cam]
        else:
            # Propagate relevance through the single tensor
            cam = self.fn.relprop(cam, **kwargs)
            return self.norm.relprop(cam, **kwargs)


#========================================================================================#
#============================ Text Embedding and Encoder ================================#
#========================================================================================#

class TextEmbed(nn.Module):
    """
    A PyTorch module for embedding textual data using a pre-defined text encoder and linear projection.
    This module encodes text using a tokenization method from TikToken, then projects the tokenized text 
    into an embedding space of specified dimension.

    Attributes:
        device (str): Device where tensors will be processed, either 'cuda' or 'cpu'.
        config (dict): Configuration dictionary specifying embedding dimensions and dropout rate.
        TextEncoder (tiktoken.Encoding): Pre-trained text encoding model (e.g., GPT-2).
        proj (nn.Linear): Linear layer for projecting tokenized text to embedding space.
        cls_token (nn.Parameter): Learnable parameter acting as a classification token.
        dropout (nn.Dropout): Dropout layer for regularization in the embedding output.
    """

    def __init__(self, 
                 config: Union[Dict], 
                 device=None):
        """
        Initializes the TextEmbed module with a specific configuration and device.
        
        Args:
            config (Union[Dict]): Configuration dictionary that must include 'embed_dim' and 'proj_dropout'.
            device (str, optional): Specifies the device ('cuda' or 'cpu'). If None, will use CUDA if available.
        """
        super().__init__()
        # Default to CUDA if available, otherwise use CPU.
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.config = config
        
        # Initialize the text encoder with a predefined model from TikToken.
        self.TextEncoder = tiktoken.get_encoding('gpt2')
        
        # Linear projection layer from token to embedding dimension.
        self.proj = Linear(1, config['embed_dim'])
        # Classification token, learnable, initialized to zeros.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config['embed_dim']))
        # Dropout layer for regularization.
        self.dropout = Dropout(config['proj_dropout'])

    def forward(self, texts):
        """
        Forward pass for converting a list of text strings into embedded representation.

        Args:
            texts (List[str]): List of text strings to encode and embed.

        Returns:
            torch.Tensor: Embedded representations of input texts.
        """
        # Encode texts using the TextEncoder, resulting in a list of numeric token lists.
        encoded_texts = [self.TextEncoder.encode(text) for text in texts]
        # Find the maximum length of encoded text to standardize the length.
        max_length = max(len(text) for text in encoded_texts)
        # Pad shorter texts to the maximum length.
        padded_texts = [text + [0] * (max_length - len(text)) for text in encoded_texts]
        # Convert list of texts to a tensor and move to the specified device.
        texts_tensor = torch.tensor(padded_texts, dtype=torch.float32).to(self.device)
        # Add a dimension to align with embedding space requirements.
        texts_tensor = torch.unsqueeze(texts_tensor, dim=-1)
        # Batch size: number of texts.
        B = texts_tensor.shape[0]
        # Expand cls_token to match the batch size.
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # Project tokenized texts to the embedding dimension.
        texts_tensor = self.proj(texts_tensor)
        # Concatenate cls_token to the beginning of each projected text.
        texts_tensor = torch.cat((cls_tokens, texts_tensor), dim=1)
        # Apply dropout to the concatenated tensor.
        embeddings = self.dropout(texts_tensor)
        return embeddings, encoded_texts
    

    def decode(self, tokens: List[int]):
        """
        Decodes a list of token IDs back to the corresponding text string.

        Args:
            tokens (List[int]): List of token IDs to decode.

        Returns:
            str: Decoded text string.
        """
        return self.TextEncoder.decode(tokens)
    

    def relprop(self, cam, **kwargs):
        """
        Method for relevance propagation to interpret the contributions of inputs to outputs.
        
        Args:
            cam (torch.Tensor): Gradients or relevance scores from higher layers.
        
        Returns:
            torch.Tensor: Relevance scores propagated back to the input layer.
        """
        # Propagate relevance through the dropout layer.
        cam = self.dropout.relprop(cam, **kwargs)
        # Separate cls_token relevance and text relevance.
        cam_cls, cam_texts = cam[:, :1, :], cam[:, 1:, :]
        # Propagate relevance through the projection layer.
        cam_texts = self.proj.relprop(cam_texts, **kwargs)
        return cam_texts
    
class TextAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5


        self.matmul1 = einsum('bhid,bhjd->bhij')
        self.matmul2 = einsum('bhij,bhjd->bhid')

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x, g):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        self.save_v(v)
        dots = self.matmul1([q, k]) * self.scale
        attn = self.softmax(dots)
        attn = self.attn_drop(attn)

        attn.requires_grad_(True)
        self.save_attn(attn)
        if g: 
            attn.register_hook(self.save_attn_gradients)

        out = self.matmul2([attn, v])
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.proj(out)
        out = self.proj_drop(out)

        return out#, attn

    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)


        (cam1, cam_v)= self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

        return self.qkv.relprop(cam_qkv, **kwargs)

class TextEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = TextAttention(dim, num_heads=heads, qkv_bias=False, attn_drop=dropout, proj_drop=dropout)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mult)
        self.mlp = Mlp(in_features = dim, hidden_features = mlp_hidden_dim, out_features = dim, drop = dropout) 

        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, text, g):
        x1, x2 = self.clone1(text, 2)
        text = self.attn(x2, g)
        text = self.add1([x1, text])
        x1, x2 = self.clone2(text, 2)
        text = self.add2([x1, self.mlp(self.norm2(x2))])
        return text

    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam

#========================================================================================#
#========================================================================================#
#========================================================================================#

class AccImgEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, config, index):
        super().__init__()
        img_size = _pair(config.img_size)
        patch_size = _pair(config.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (index)
        
        self.num_patches = num_patches
        self.proj = Conv2d(config.in_channels, config.embed_dim, kernel_size=patch_size, stride=patch_size) 
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches+1, config.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))   
        self.dropout = Dropout(config.proj_dropout)
        self.add = Add()

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = rearrange(x, 'b c w h t -> (b t) c w h')
        x = self.proj(x)  
        x = x.flatten(2).transpose(1, 2)  

        x = rearrange(x, '(b t) n e -> b (t n) e', b=B)  


        embeddings = torch.cat((cls_tokens, x), dim=1) 
        embeddings = self.add([embeddings , self.position_embeddings])
        embeddings = self.dropout(embeddings)

        return embeddings
    
    def relprop(self, cam, **kwargs):
        cam = self.dropout.relprop(cam, **kwargs)
        (cam_patches, _) = self.add.relprop(cam, **kwargs)
        cam_patches = cam_patches[:, 1:, :]

        cam_patches = rearrange(cam_patches, 'b (t n) e -> (b t) n e', n=1) 

        cam_patches = cam_patches.transpose(1, 2).flatten(0, 1)
        cam_patches = cam_patches.reshape(cam_patches.shape[0], cam_patches.shape[1],
                                        (self.img_size[0] // self.patch_size[0]), 
                                        (self.img_size[1] // self.patch_size[1]))
        
        # Propagate through projection
        cam_patches = self.proj.relprop(cam_patches, **kwargs)
        cam_patches = rearrange(cam_patches, '(b t) c w h -> b c w h t', n=1) 
        return cam_patches

class SpatialMetAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5


        self.matmul1 = einsum('bhid, bhjd->bhij')
        self.matmul12 = einsum('bhid, bhjd->bhij')
        self.matmul2 = einsum('bhij, bhjd->bhid')

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)


        self.add1 = Add()

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x, met, g):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        self.save_v(v)
        dots = self.matmul1([q, k]) * self.scale

        if met is not None:
            met_qkv = self.qkv(met)
            met_q, met_k, met_v = rearrange(met_qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)
            self.met_dots = self.matmul12([met_q, met_k]) * self.scale
            dots = self.add1([dots, self.met_dots])

        self.met = met

        attn = self.softmax(dots)
        attn = self.attn_drop(attn)

        # attn.requires_grad_(True)
        self.save_attn(attn)
        # if g: 
        attn.register_hook(self.save_attn_gradients)

        out = self.matmul2([attn, v])
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.proj(out)
        out = self.proj_drop(out)

        return out#, attn

    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

        (cam1, cam_v)= self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        if self.met is not None: 
            (cam1, cam_met) = self.add2.relprop(cam1, **kwargs)
            (cam_q_met, cam_k_met) = self.matmul12.relprop(cam_met, **kwargs)
            cam_q_met /= 2
            cam_k_met /= 2
            cam_qkv_met = rearrange([cam_q_met, cam_k_met, cam_met], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)
            out_met = self.qkv.relprop(cam_qkv, **kwargs)

        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

        return self.qkv.relprop(cam_qkv, **kwargs)#, out_met
    
class SpatialMetEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = SpatialMetAttention(dim, num_heads = heads, qkv_bias=False, attn_drop=dropout, proj_drop=dropout) 
        self.norm2 = LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mult)
        self.mlp = Mlp(in_features = dim, hidden_features = mlp_hidden_dim, out_features = dim, drop = dropout) 

        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()
    
    def forward(self, x, met=None, g = True):
        x1, x2 = self.clone1(x, 2)
        x = self.add1([x1, self.attn(self.norm1(x2), self.norm1(met), g)])

        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.mlp(self.norm2(x2))])

        return x
    
    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2, cam_met = cam2
        cam_met = self.norm1.relprop(cam_met, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)

        return cam#, cam_met
#========================================================================================#
#========================================================================================#
#========================================================================================#
class AccMetEmbed(nn.Module):
    """ Metadata to Patch Embedding """
    def __init__(self, config, index):
        super().__init__()
        num_patches = 4 * index  # Make sure index is used correctly
        self.num_patches = num_patches
        self.proj = Linear(1, config.embed_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.dropout = Dropout(config.proj_dropout)
        self.add = Add()


    def forward(self, met):
        B = met.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        met = rearrange(met, 'b c t d -> b t (c d)')
        met = met.unsqueeze(-1)  # More explicit dimension expansion
        met = self.proj(met)
        met = rearrange(met, 'b t n e -> b (t n) e')

        embeddings = torch.cat((cls_tokens, met), dim=1) 
        embeddings = self.add([embeddings , self.position_embeddings])
        embeddings = self.dropout(embeddings)
        return embeddings
    
    def relprop(self, cam, **kwargs):
        # Propagate relevance through dropout
        cam = self.dropout.relprop(cam, **kwargs)
        (cam_patches, _) = self.add.relprop(cam, **kwargs)

        cam_patches = cam_patches[:, 1:, :]

        # Reshape back to match output from proj
        cam_patches = rearrange(cam_patches, 'b (t n) e -> b t n e', n=1)  
        cam_patches = self.proj.relprop(cam_patches.squeeze(-1), **kwargs)  

        cam_patches = cam_patches.squeeze(-1)
        cam_patches = rearrange(cam_patches, 'b t (c d)-> b c t d', n=1)  

        return cam_patches
#========================================================================================#
#========================================================================================#
#========================================================================================#

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0. ):
        super().__init__()


        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.matmul1 = einsum('bid, bjd -> bij')
        self.matmul2 = einsum('bij, bjd -> bid')

        # self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.to_q = Linear(query_dim, inner_dim, bias=False)
        self.to_k = Linear(context_dim, inner_dim, bias=False)
        self.to_v = Linear(context_dim, inner_dim, bias=False)


        self.attn_drop = Dropout(dropout)
        self.proj = Linear(query_dim, query_dim)
        self.proj_drop = Dropout(dropout)
        self.softmax = Softmax(dim=-1)


        self.add1 = Add()

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self,  x, context=None, g = True):

        b, n, _, h = *x.shape, self.heads
        # qkv = self.qkv(x)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        self.save_v(v)

        dots = self.matmul1([q, k]) * self.scale
        attn = self.softmax(dots)
        attn = self.attn_drop(attn)

        # attn.requires_grad_(True)
        self.save_attn(attn)
        # if g: 
        attn.register_hook(self.save_attn_gradients)

        out = self.matmul2([attn, v])
        # out = rearrange(out, 'b h n d -> b n (h d)')
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.proj(out)
        out = self.proj_drop(out)

        
        return out#, attn

    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

        (cam1, cam_v)= self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        # cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)
        cam_q, cam_k, cam_v = map(lambda t: rearrange(t, '(b h) n d -> b n (h d)', h=self.num_heads), (cam_q, cam_k, cam_v))
        cam_qkv = torch.cat([cam_q, cam_k, cam_v], dim=-1)

        return self.qkv.relprop(cam_qkv, **kwargs)
    
class MultiModalEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, context_dim=9, mult=4, dropout=0.):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = CrossAttention(dim, context_dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mult)
        self.mlp = Mlp(in_features = dim, hidden_features = mlp_hidden_dim, out_features = dim, drop = dropout) 

        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, x, context=None, g = True):
        x1, x2 = self.clone2(x, 2)
        x = self.add1([x1, self.attn(self.norm1(x2), context, g)])
        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        return x
    
    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2, cam_contxt = cam2
        cam_contxt = self.norm1.relprop(cam_contxt, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)

        return cam#, cam_contxt
    
  