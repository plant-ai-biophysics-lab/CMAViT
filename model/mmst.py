import torch
from torch import nn
from timm.models.layers import trunc_normal_
from ml_collections import ConfigDict
# from torch.jit import script
from typing import Dict, List, NamedTuple, Optional, Tuple, Union, cast, overload

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from model import configs, engine
seed = 1987 + engine.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)

from model.attention import TextEmbed, AccMetEmbed, AccImgEmbed, MultiRegressionHead
from model.attention import TextEncoder, SpatialMetEncoder, MultiModalTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

class MMST_ViT(nn.Module):
    def __init__(self, config: Union[Dict], 
                 cond=False): #
        super().__init__()


        if not isinstance(config, ConfigDict):
            raise ValueError("Config must be an instance of ml_collections.ConfigDict.")
                
        assert config.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.cond = cond
        self.mask_modality = config.mask_modality


        if self.mask_modality != 'text':
            self.text_embed = TextEmbed(config)
            self.text_transformer = TextEncoder(config.embed_dim, 
                            config.num_layers, 
                            config.num_heads, 
                            dim_head = 96, 
                            mult=4, 
                            dropout=config.proj_dropout)

        self.img_embeds = nn.ModuleList([AccImgEmbed(config, i).to(device) for i in range(1, 16)])
        self.met_embeds = nn.ModuleList([AccMetEmbed(config, i).to(device) for i in range(1, 16)]) 

        self.spatialmet_encoder = SpatialMetEncoder(
            config.embed_dim,
            config.num_layers, 
            config.num_heads, 
            dim_head = 96, 
            mult=4, 
            dropout=config.proj_dropout)
        
        self.cross_attn_encoder = MultiModalTransformer(
            config.embed_dim, 
            config.num_layers, 
            config.num_heads, 
            dim_head = 96, 
            context_dim=config.embed_dim, 
            mult=4,  
            dropout=config.proj_dropout)
        
        self.pool = config.pool
        self.head = MultiRegressionHead(config)

    #     self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.1)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
            

    
    def process_embedding(self, 
                          x: torch.Tensor, 
                          s_embeds: nn.Module, 
                          met: torch.Tensor,  
                          m_embeds: nn.Module, 
                          encoder) -> torch.Tensor:
        
        x_s = s_embeds(x)
        x_m = m_embeds(met)
        x_o, attn = encoder(x_s, x_m)
    
        return x_o, attn

    def forward(self, 
                img: torch.Tensor,
                context: str = None, 
                met: torch.Tensor = None, 
                yz: torch.Tensor = None): 
        
        # Conditional input modulation
        if self.cond:
            img = img * yz

        out, attns = [], {}


        if self.mask_modality != 'text':
            context = self.text_embed(context)
            context, text_attn = self.text_transformer(context)
            context = context.mean(dim=1)
            context = torch.unsqueeze(context, dim=1)
            attns['ItextAttn'] = [text_attn]

        for index, (img_embed, met_embed) in enumerate(zip(self.img_embeds, self.met_embeds)):
            if self.mask_modality != 'image':
                img_t = img[..., :index+1]
            else:
                img_t = torch.zeros_like(img[..., :index+1])

            if self.mask_modality != 'met':
                met_t = met[:, :, :index+1, :]
            else:
                met_t = torch.zeros_like(met[:, :, :index+1, :])

            ImgMet_t, ImgMetAttn = self.process_embedding(img_t, 
                                             img_embed, 
                                             met_t, 
                                             met_embed, 
                                             self.spatialmet_encoder)
            ImgMet_t_mean = ImgMet_t.mean(dim=1)
           
            if self.mask_modality != 'text':
                ImgMet_t = torch.unsqueeze(ImgMet_t_mean, dim = 1)
                ImgMetText_t, MMAttn = self.cross_attn_encoder(ImgMet_t, context)
                ImgMetText_t = ImgMetText_t.mean(dim=1)
                ImgMetText_t = torch.unsqueeze(ImgMetText_t, dim = 1)
                ImgMetText_t = torch.cat((ImgMetText_t, ImgMet_t), dim = 1)
                ImgMetText_t = ImgMetText_t.view(ImgMetText_t.size(0), -1)
                out.append(ImgMetText_t)
            else:
                out.append(ImgMet_t_mean)

            # key_imgmet = f'ImgMetAttn_{index}'
            # key_mmattn = f'MMAttn_{index}'
            # if key_imgmet not in attns:
            #     attns[key_imgmet] = []
            
            #     attns[key_mmattn] = []

            # attns[key_imgmet].append(ImgMetAttn)
            # attns[key_mmattn].append(MMAttn)
        
        preds, logits = self.head(out)

        return preds, logits , attns




