import torch
import argparse
import numpy as np
# Import custom dataloader, model, and engine modules
from src import dataloader
from model import configs, engine


# from model.DualEncoder import ViT, CrossAttnViT
# from model.mmt import MMTR
from model.mmst import MMST_ViT
from model.mmst_lrp import MMST_ViT_LRP

def main(args):
    # fix the seed for reproducibility
    seed = 1987 + engine.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Check if GPU is available, set device accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Check if there is GPU(s): {torch.cuda.is_available()}")

    # Extract hyperparameters from command line arguments
    Exp_name = args.exp_name
    batch_size = args.batch_size
    embd_size = args.embd_size
    num_heads = args.num_heads
    num_layers = args.num_layers
    in_channels = args.in_channels
    dropout = args.dropout
    post_norm = args.postnorm
    lr = args.lr
    wd = args.wd
    loss = args.loss
    epochs = args.epochs
    resampling = args.resampling
    cond_status = args.cond
    tokenizer = args.tokenizer
    mask_modality = args.mask_modality


    # Get data loaders from custom dataloader module
    data_loader_training, data_loader_validate, data_loader_test = dataloader.dataloaders(
        batch_size = batch_size, 
        img_size = 16,
        in_channels = in_channels, 
        resmapling_status = resampling,
        data = 's2',
        exp_name = Exp_name
        )


    # # Create model configuration using custom configs module
    config = configs.Configs(
        img_size = 16, 
        patch_size = 8, 
        embed_dim = embd_size, 
        mlp_dim = 512, 
        pool = 'cls',
        in_channels = in_channels,
        out_channels = 1, 
        num_heads = num_heads, 
        num_layers = num_layers, 
        cond = cond_status,
        multi_conv = False,
        attn_dropout = dropout, 
        proj_dropout = dropout, 
        drop_path = 0.0,
        post_norm = post_norm, 
        vis = True, 
        tokenizer = tokenizer,
        mask_modality = mask_modality
        ).call()
    
    # # Create and move model to GPU
    model = MMST_ViT(config, cond = False).to(device)
    # model = MMST_ViT_LRP(config, cond = False).to(device)

    # # Calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    # # Create an instance of ImbYieldEst engine
    YE = engine.ViTYieldEst(model, 
                                 lr = lr, 
                                 wd = wd, 
                                 exp = Exp_name)
    # # Train the model
    _ = YE.train(data_loader_training, data_loader_validate, 
                      loss = loss, 
                      epochs = epochs, 
                      loss_stop_tolerance = 20)

    # # Predict 
    model = MMST_ViT_LRP(config, cond = False).to(device)
    _ = YE.predict(model, data_loader_training, category= 'train', iter = 1)
    _ = YE.predict(model, data_loader_validate, category= 'valid', iter = 1)
    _ = YE.predict(model, data_loader_test, category= 'test', iter = 1)


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Imbalance Deep Yield Estimation")
    parser.add_argument("--exp_name",    type=str,   default = "test",help = "Experiment name")
    parser.add_argument("--batch_size",  type=int,   default = 32,   help = "Batch size")
    parser.add_argument("--embd_size",   type=int,   default = 768,  help = "Embedding size")
    parser.add_argument("--num_heads",   type=int,   default = 8,     help = "Number of attention heads")
    parser.add_argument("--num_layers",  type=int,   default = 6,     help = "Number of transformer layers")
    parser.add_argument("--in_channels", type=int,   default = 8,     help = "Number of input channels")
    parser.add_argument("--dropout",     type=float, default = 0.1,   help = "Amount of dropout")
    parser.add_argument("--postnorm",    type=str,   default = False, help = "Post or Before Normalization for Self-Attention")
    parser.add_argument("--lr",          type=float, default = 0.0001, help = "Learning rate")
    parser.add_argument("--wd",          type=float, default = 0.0001,  help = "Value of weight decay")
    parser.add_argument("--epochs",      type=int,   default = 100,   help = "The number of epochs")
    parser.add_argument("--loss",        type=str,   default = "mse", help = "Loss function  mse wmse huber wass")
    parser.add_argument("--resampling",  type=str,   default = False, help = "Weight resampling status") 
    parser.add_argument("--cond",        type=str,   default = False, help = "Conditional Model to use") 
    parser.add_argument("--tokenizer",   type=str,   default = 'EC', help = "tokenizer strategy including EC, HA, CAC") 
    parser.add_argument("--mask_modality", type=str,   default = None, help = "mask modality name inclding text, met and img")

    args = parser.parse_args()

    main(args)


