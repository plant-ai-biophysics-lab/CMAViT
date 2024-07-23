import torch
from torch import nn
import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
from ml_collections import ConfigDict
from typing import Dict, Union
torch.autograd.set_detect_anomaly(True)

# from model import configs, engine
from model.attn_lrp import TextEmbed, AccMetEmbed, AccImgEmbed, MultiRegressionHead
from model.attn_lrp import TextEncoder, SpatialMetEncoder, MultiModalEncoder
from model.layers_ours import *



seed = 1987 #+ engine.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention


class MMST_ViT_LRP(nn.Module):
    def __init__(self, config: Union[Dict], 
                 cond=False): #
        super().__init__()

        if not isinstance(config, ConfigDict):
            raise ValueError("Config must be an instance of ml_collections.ConfigDict.")
        
        assert config.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.cond = cond
        self.mask_modality = config.mask_modality

        self.pool = IndexSelect()
        self.add = Add()


        if self.mask_modality != 'text':
            self.text_embed = TextEmbed(config)
            self.text_transformer = nn.ModuleList([TextEncoder(config.embed_dim, 
                            config.num_layers, 
                            config.num_heads, 
                            dim_head = 96, 
                            mult=4, 
                            dropout=config.proj_dropout)
                            for i in range(config.num_layers)])

        self.img_embeds = nn.ModuleList([AccImgEmbed(config, i).to(device) for i in range(1, 16)])
        self.met_embeds = nn.ModuleList([AccMetEmbed(config, i).to(device) for i in range(1, 16)]) 

        self.spatialmet_encoder = nn.ModuleList([SpatialMetEncoder(
            config.embed_dim,
            config.num_layers, 
            config.num_heads, 
            dim_head = 96, 
            mult=4, 
            dropout=config.proj_dropout)
            for i in range(config.num_layers)])
        
        self.cross_attn_encoder = nn.ModuleList([MultiModalEncoder(
            config.embed_dim, 
            config.num_layers, 
            config.num_heads, 
            dim_head = 96, 
            context_dim=config.embed_dim, 
            mult=4,  
            dropout=config.proj_dropout)
            for i in range(config.num_layers)])
        
        
        self.head = MultiRegressionHead(config)

        self.inp_grad = None

    def save_inp_grad(self,grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad


    def forward(self, 
                img: torch.Tensor,
                context: str = None, 
                met: torch.Tensor = None, 
                yz: torch.Tensor = None,
                g = True): 
        
        # Conditional input modulation
        if self.cond:
            img = img * yz

        weekly_preds = []

        context, tokens = self.text_embed(context)
        context.register_hook(self.save_inp_grad)


        for txt_blk in self.text_transformer:
            context = txt_blk(context, g) 
        # context = self.pool(context, dim=1, indices=torch.tensor(0, device=context.device))
        context = context.mean(dim=1)
        context = torch.unsqueeze(context, dim=1)


        for index, (img_embed, met_embed) in enumerate(zip(self.img_embeds, self.met_embeds)):

            img_w = img[..., :index+1]
            met_w = met[:, :, :index+1, :]
            img_w = img_embed(img_w)
            met_w = met_embed(met_w)
            img_w.register_hook(self.save_inp_grad)
            met_w.register_hook(self.save_inp_grad)


            for imblk in self.spatialmet_encoder:
                imgmet_w = imblk(img_w, met_w, g)

            # imgmet_w_mean = self.pool(imgmet_w, dim=1, indices=torch.tensor(0, device=imgmet_w.device))
            imgmet_w_mean = imgmet_w.mean(dim=1)
            imgmet_w_mean = torch.unsqueeze(imgmet_w_mean, dim = 1)


            for mitblk in self.cross_attn_encoder:
                ImgMetText_w = mitblk(imgmet_w_mean, context, g) 

            # ImgMetText_w = self.pool(ImgMetText_w, dim=1, indices=torch.tensor(0, device=ImgMetText_w.device))
            ImgMetText_w = ImgMetText_w.mean(dim=1)
            ImgMetText_w = torch.unsqueeze(ImgMetText_w, dim = 1)
            ImgMetText_w = self.add([ImgMetText_w , imgmet_w_mean])
            weekly_preds.append(ImgMetText_w[:, 0])

        preds = self.head(weekly_preds)

        return preds
    
    def relprop(self, cam_list, **kwargs):
        relprops = []
        cams = self.head.relprop(cam_list)  # Assuming head.relprop expects a list

        for cam, img_embed, met_embed in zip(cams, self.img_embeds, self.met_embeds):
            # Propagate relevance through the regression head
            print("conservation 1", cam.sum())

            # (cam_img_met_text, cam_img_met) = self.add.relprop(cam, **kwargs)
            # cam_img_met_text = cam_img_met_text.squeeze(1)
            # cam_img_met_text = self.pool.relprop(cam_img_met_text, **kwargs)

            cams_img_met_text = []
            for mitblk in self.cross_attn_encoder:
                grad = mitblk.attn.get_attn_gradients()
                cam = mitblk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams_img_met_text.append(cam.unsqueeze(0))

            rollout = compute_rollout_attention(cams_img_met_text, start_layer=0)
            cam02 = rollout[:, 0, 1:]


            cam_img_met = cam_img_met.squeeze(1)
            cam_img_met = self.pool.relprop(cam_img_met, **kwargs)

            cams_img_met = []
            for imblk in self.spatialmet_encoder:
                grad01 = imblk.attn.get_attn_gradients()
                cam01= imblk.attn.get_attn_cam()
                cam01 = cam01[0].reshape(-1, cam01.shape[-1], cam01.shape[-1])
                grad01 = grad01[0].reshape(-1, grad01.shape[-1], grad01.shape[-1])
                cam01 = grad01 * cam01
                cam01 = cam01.clamp(min=0).mean(dim=0)
                cams_img_met.append(cam01.unsqueeze(0))
            rollout01 = compute_rollout_attention(cams_img_met, start_layer=0)
            cam01 = rollout01[:, 0, 1:]

            # cam_text = cam_img_met.squeeze(1)
            # cam_img_met = self.pool.relprop(cam_img_met, **kwargs)

            cams_text = []
            for txt_blk in self.text_transformer:
                grad00 = txt_blk.attn.get_attn_gradients()
                cam00= txt_blk.attn.get_attn_cam()
                cam00 = cam00[0].reshape(-1, cam00.shape[-1], cam00.shape[-1])
                grad00 = grad00[0].reshape(-1, grad00.shape[-1], grad00.shape[-1])
                cam00 = grad00 * cam00
                cam00 = cam00.clamp(min=0).mean(dim=0)
                cams_text.append(cam00.unsqueeze(0))
            rollout00 = compute_rollout_attention(cams_text, start_layer=0)
            cam00 = rollout00[:, 0, 1:]

        return cam00, cam01, cam02
    




