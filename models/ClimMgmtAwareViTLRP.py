import torch
from torch import nn
from ml_collections import ConfigDict
from typing import Dict, Union
import os 
from timm.models.layers import trunc_normal_
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from models.configs import set_seed
set_seed(1987)

from models.lrp_attention import AccMetEmbed, AccImgEmbed, MultiRegressionHead, SpatialMetEncoder, TextEmbed, TextEncoder, MultiModalTransformer
from models.lrp_attention import compute_rollout_attention


device = "cuda" if torch.cuda.is_available() else "cpu"

class ClimMgmtAware_ViT(nn.Module):
    def __init__(self, config: Union[Dict]): #
        super().__init__()

        if not isinstance(config, ConfigDict):
            raise ValueError("Config must be an instance of ml_collections.ConfigDict.")
                
        self.text_embed = TextEmbed(model_name="distilbert-base-uncased", max_length=250)
        self.text_transformer = nn.ModuleList([TextEncoder(config.context_dim, 
            config.num_layers, 
            config.num_heads, 
            dim_head = 8, 
            mult=4, 
            dropout=config.proj_dropout)
            for i in range(config.num_layers)])

        self.img_embeds = AccImgEmbed(config, 15).to(device)
        self.met_embeds = AccMetEmbed(config, 15).to(device)

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
            context_dim=config.context_dim, 
            mult=4,  
            dropout=config.proj_dropout)
        
        self.head = MultiRegressionHead(config)

        self.apply(self._init_weights)
        self.inp_grad = None

    def save_inp_grad(self,grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad

    def _init_weights(self, m):
        # Initialize Linear layers
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = 0.01)  # Truncated normal initialization
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Initialize bias to zero

        # Initialize Conv2d layers
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaiming initialization for Conv2d
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Initialize bias to zero

        # Initialize LayerNorm layers
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  # Set bias to zero
            nn.init.constant_(m.weight, 1.0)  # Set weights to 1

        # Initialize BatchNorm2d layers (if used)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)  # Set weights to 1 for BatchNorm
            nn.init.constant_(m.bias, 0)  # Set bias to zero for BatchNorm

        # # Optionally, initialize Embedding layers
        # elif isinstance(m, nn.Embedding):
        #     nn.init.normal_(m.weight, mean=0, std=0.01)  # Normal initialization for embeddings

    def forward(self, 
                img: torch.Tensor,
                context: str = None, 
                met: torch.Tensor = None, 
                yz: torch.Tensor = None, 
                cond: str = False): 
        context, mask = self.text_embed(context)
        context.register_hook(self.save_inp_grad)

        
        for txt_blk in self.text_transformer:
            context = txt_blk(context, g = True) 
            
            
        context = context.mean(dim=1)
        context = torch.unsqueeze(context, dim=1)
    
        img = self.img_embeds(img)
        met = self.met_embeds(met)
        ImgMet_t = self.spatialmet_encoder(x = img, met = met)

        ImgMet_t_mean = ImgMet_t[:, 1:].mean(dim=1)
        ImgMet_t_mean = torch.unsqueeze(ImgMet_t_mean, dim = 1)
        ImgMetText_t, _ = self.cross_attn_encoder(ImgMet_t_mean, context)
        ImgMetText_t = ImgMetText_t.mean(dim=1)
        ImgMetText_t = torch.unsqueeze(ImgMetText_t, dim = 1)
        ImgMetText_t = torch.cat((ImgMetText_t, ImgMet_t_mean), dim = 1)
        ImgMetText_t = ImgMetText_t.view(ImgMetText_t.size(0), -1)
        preds = self.head([ImgMetText_t])

        return preds


    def relprop(self, cam=None,method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        # print(kwargs)
        # print("conservation 1", cam.sum())
        cam = self.head.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        cam = self.pool.relprop(cam, **kwargs)
        cam = self.norm.relprop(cam, **kwargs)
        for blk in reversed(self.blocks):
            cam = blk.relprop(cam, **kwargs)

        # print("conservation 2", cam.sum())
        # print("min", cam.min())

        if method == "full":
            (cam, _) = self.add.relprop(cam, **kwargs)
            cam = cam[:, 1:]
            cam = self.patch_embed.relprop(cam, **kwargs)
            # sum on channels
            cam = cam.sum(dim=1)
            return cam

        elif method == "rollout":
            # cam rollout
            attn_cams = []
            for blk in self.blocks:
                attn_heads = blk.attn.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            cam = cam[:, 0, 1:]
            return cam
        
        # our method, method name grad is legacy
        elif method == "transformer_attribution" or method == "grad":
            cams = []
            for blk in self.blocks:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = rollout[:, 0, 1:]
            return cam
            
        elif method == "last_layer":
            cam = self.blocks[-1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[-1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "last_layer_attn":
            cam = self.blocks[-1].attn.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "second_layer":
            cam = self.blocks[1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

