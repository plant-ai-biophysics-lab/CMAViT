import torch
from torch import nn
from ml_collections import ConfigDict
from typing import Dict, Union
from einops import rearrange, repeat
import os 
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import random
seed = 1987
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from models.attention import TextEmbed, TextEmbedBERT, AccMetEmbed, AccImgEmbed, AccImgPredEmbed, MultiRegressionHead, YzEmbed, SingleRegressionHead
from models.attention import TextEncoder, SpatialMetEncoder, MultiModalTransformer, SpatialMetYzEncoder 

device = "cuda" if torch.cuda.is_available() else "cpu"


class MMST_ViT(nn.Module):
    def __init__(self, config: Union[Dict]): #
        super().__init__()

        if not isinstance(config, ConfigDict):
            raise ValueError("Config must be an instance of ml_collections.ConfigDict.")
                
        assert config.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.mask_modality = config.mask_modality


        if self.mask_modality != 'text':
            self.text_embed = TextEmbed(config)
            self.text_transformer = nn.ModuleList([TextEncoder(config.context_dim, 
                config.num_layers, 
                config.num_heads, 
                dim_head = 96, 
                mult=4, 
                dropout=config.proj_dropout)
                for i in range(config.num_layers)])
            


        # self.img_embeds = AccImgEmbed(config, 1).to(device)
        # self.met_embeds = AccMetEmbed(config, 1).to(device)

        self.img_embeds = nn.ModuleList([AccImgEmbed(config, i).to(device) for i in range(14, 16)])
        self.met_embeds = nn.ModuleList([AccMetEmbed(config, i).to(device) for i in range(14, 16)])

        self.spatialmet_encoder = SpatialMetEncoder(
            config.embed_dim,
            config.num_layers, 
            config.num_heads, 
            dim_head = 96, 
            mult=4, 
            dropout=config.proj_dropout)
        
        self.temporal_tokens = nn.Parameter(torch.randn(1, 1, config.embed_dim))
        self.cross_attn_encoder = MultiModalTransformer(
            config.embed_dim, 
            config.num_layers, 
            config.num_heads, 
            dim_head = 96, 
            context_dim=config.context_dim, 
            mult=4,  
            dropout=config.proj_dropout)
        
        self.pool = config.pool
        self.head = SingleRegressionHead(config)
        self.norm = nn.LayerNorm(config.embed_dim)

    #     self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.01)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def _single_week(self, img_embed, met_embed, img, context, met, pre_pred):

        # IMG and MET
        img = torch.cat((pre_pred, img), dim = 1)
        img = img_embed(img) #self.img_embeds(img)
        met = met_embed(met) #self.met_embeds(met)

        ImgMet_t, _ = self.spatialmet_encoder(img, met)
        ImgMet_t_mean = ImgMet_t[:, 1:].mean(dim=1)
        ImgMet_t_mean = torch.unsqueeze(ImgMet_t_mean, dim = 1)

        ImgMetText_t, _ = self.cross_attn_encoder(ImgMet_t_mean, context)
        ImgMetText_t = ImgMetText_t.mean(dim=1)
        ImgMetText_t = torch.unsqueeze(ImgMetText_t, dim = 1)
        ImgMetText_t = torch.cat((ImgMetText_t, ImgMet_t_mean), dim = 1)

        ImgMetText_t = ImgMetText_t.view(ImgMetText_t.size(0), -1)

        return ImgMetText_t
    
    def forward(self, 
                img: torch.Tensor,
                context: str = None, 
                met: torch.Tensor = None, 
                yz: torch.Tensor = None, 
                cond: str = False): 
        
        # Conditional input modulation
        # if bool(cond) == True:
        #     img = img * yz

        out = []
        B = img.shape[0]
        init_pred = torch.zeros(B, 1, 16, 16, 1, dtype=torch.float32).to(device)
            
        # Text
        block_attn, text_attn = [], []
        context = self.text_embed(context)
        for txt_blk in self.text_transformer:
            context, attn = txt_blk(context, g = True) 
            block_attn.append(attn)
        context = context.mean(dim=1)
        context = torch.unsqueeze(context, dim=1)
        block_attn = torch.stack(block_attn, dim=4) 
        text_attn.append(block_attn)

        # for index in range(13, 15):
        for index, (img_embed, met_embed) in enumerate(zip(self.img_embeds, self.met_embeds)):

            img_t = img[..., :index+13+1]
            met_t = met[:, :, :index+13+1, :]
            init_pred = repeat(init_pred, 'b c h w 1 -> b c h w new_index', new_index = index+14)

            this_week_pred = self._single_week(img_embed, met_embed, img_t, context, met_t, init_pred)
            this_week_pred = self.head(this_week_pred)
            init_pred = torch.unsqueeze(this_week_pred, dim = -1)
            

            out.append(this_week_pred)

        return out, text_attn

















    #     self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.1)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
            

    
    # def process_embedding(self, 
    #                       x: torch.Tensor, 
    #                       s_embeds: nn.Module, 
    #                       met: torch.Tensor,  
    #                       m_embeds: nn.Module, 
    #                       encoder) -> torch.Tensor:
        
    #     x_s = s_embeds(x)
    #     x_m = m_embeds(met)
    #     x_o, attn = encoder(x_s, x_m)
    
    #     return x_o, attn

    # def forward(self, 
    #             img: torch.Tensor,
    #             context: str = None, 
    #             met: torch.Tensor = None, 
    #             yz: torch.Tensor = None): 
        
    #     # Conditional input modulation
    #     if self.cond is True:
    #         img = img * yz

    #     out, attns = [], {}

    #     if self.mask_modality != 'text':
    #         context = self.text_embed(context)
    #         context, text_attn = self.text_transformer(context)
    #         context = context.mean(dim=1)
    #         context = torch.unsqueeze(context, dim=1)
    #         attns['ItextAttn'] = [text_attn]

    #     for index, (img_embed, met_embed) in enumerate(zip(self.img_embeds, self.met_embeds)):
    #         if self.mask_modality != 'image':
    #             img_t = img[..., :index+1]
    #         else:
    #             img_t = torch.zeros_like(img[..., :index+1])

    #         if self.mask_modality != 'met':
    #             met_t = met[:, :, :index+1, :]
    #         else:
    #             met_t = torch.zeros_like(met[:, :, :index+1, :])

    #         ImgMet_t, ImgMetAttn = self.process_embedding(img_t, 
    #                                          img_embed, 
    #                                          met_t, 
    #                                          met_embed, 
    #                                          self.spatialmet_encoder)
    #         ImgMet_t_mean = ImgMet_t.mean(dim=1)
           
    #         if self.mask_modality != 'text':
    #             ImgMet_t_mean = torch.unsqueeze(ImgMet_t_mean, dim = 1)
    #             ImgMetText_t, MMAttn = self.cross_attn_encoder(ImgMet_t_mean, context)
    #             ImgMetText_t = ImgMetText_t.mean(dim=1)
    #             ImgMetText_t = torch.unsqueeze(ImgMetText_t, dim = 1)
    #             ImgMetText_t = torch.cat((ImgMetText_t, ImgMet_t_mean), dim = 1)
    #             ImgMetText_t = ImgMetText_t.view(ImgMetText_t.size(0), -1)
    #             out.append(ImgMetText_t)
    #         else:
    #             out.append(ImgMet_t_mean)

    #         # key_imgmet = f'ImgMetAttn_{index}'
    #         # key_mmattn = f'MMAttn_{index}'
    #         # if key_imgmet not in attns:
    #         #     attns[key_imgmet] = []
            
    #         #     attns[key_mmattn] = []

    #         # attns[key_imgmet].append(ImgMetAttn)
    #         # attns[key_mmattn].append(MMAttn)
        
    #     preds = self.head(out)

    #     return preds, attns