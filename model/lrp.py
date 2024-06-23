import argparse
import torch
import numpy as np
from numpy import *

# compute rollout between attention layers
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention

# class LRP:
#     def __init__(self, model):
#         self.model = model
#         self.model.eval()

#     def generate_LRP(self, img, text, met, yz, index=None, method="transformer_attribution", is_ablation=False, start_layer=0):
#         output = self.model(img, text, met, yz)
        

#         kwargs = {"alpha": 1}
#         if index == None:
#             index = np.argmax(output.cpu().data.numpy(), axis=-1)

#         one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
#         one_hot[0, index] = 1
#         one_hot_vector = one_hot
#         one_hot = torch.from_numpy(one_hot).requires_grad_(True)
#         one_hot = torch.sum(one_hot.cuda(0) * output)

#         self.model.zero_grad()
#         one_hot.backward(retain_graph=True)

#         return self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method=method, is_ablation=is_ablation,
#                                   start_layer=start_layer, **kwargs)



class LRP:
    def __init__(self, model):
        self.model = model
        # self.model.eval()  # Ensure the model is in evaluation mode

    def generate_LRP(self, img, text, met, yz):
        # Generate model output, assuming output is a list of tensors
        outputs = self.model(img, context=text, met=met, yz=yz)

        # Initialize a list to hold the gradients (initial CAMs) for each output
        initial_cams = []

        # Loop through each output in the list
        for i, output in enumerate(outputs):
            output.requires_grad_(True)  # Enable gradient computation for this output
            self.model.zero_grad()  # Clear previous gradients

            # Create a gradient tensor that matches the shape of the current output
            gradient_tensor = torch.ones_like(output, requires_grad=True)
            # Use retain_graph=True if this is not the last output to process
            output.backward(gradient_tensor, retain_graph = (i < len(outputs) - 1))

            # Ensure that gradients are computed
            if output.grad is not None:
                # Retrieve the gradient of the output as the initial cam for relprop
                initial_cams.append(output.grad.detach())  # Detach to prevent further gradient computation
            else:
                # Handle cases where gradient could not be computed
                print(f"Warning: No gradient for output {i}, skipping relprop.")
                initial_cams.append(None)

            # Optionally clear the gradients to free memory
            # output.grad = None

        # Call the relevance propagation function of the model with the list of all initial cams
        if any(cam is not None for cam in initial_cams):  # Proceed if there's at least one non-None gradient
            relevance_scores = self.model.relprop(initial_cams, img=img, context=text, met=met, yz=yz)
        else:
            relevance_scores = [None] * len(outputs)

        return relevance_scores


class Baselines:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_cam_attn(self, input, index=None):
        output = self.model(input.cuda(1), register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda(1) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        #################### attn
        grad = self.model.blocks[-1].attn.get_attn_gradients()
        cam = self.model.blocks[-1].attn.get_attention_map()
        cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam
        #################### attn

    def generate_rollout(self, input, start_layer=0):
        self.model(input)
        blocks = self.model.blocks
        all_layer_attentions = []
        for blk in blocks:
            attn_heads = blk.attn.get_attention_map()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        return rollout[:,0, 1:]