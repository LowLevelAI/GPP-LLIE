# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import math
from model_incontext_revise import DiT_incontext_revise
from diffusion import create_diffusion
from vae.autoencoder import AutoencoderKL
from vae.cond_encoder import CondEncoder
import options.options as option
from LoL_dataset import LoL_Dataset_RIDCP, create_dataloader
from utils import util
from torchvision.utils import save_image
from download import load_model

from torch.nn import functional as F


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    opt = option.parse(args.opt, is_train=True)
    #### distributed training settings
    opt['dist'] = False
    rank = -1
    print('Disabled distributed training.')
    device = torch.device('cuda:0')
    
    # Setup an experiment folder:
    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
    
        util.mkdir_and_rename(
            opt['path']['experiments_root'])  # rename experiment folder if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                        and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))

    
    model = DiT_incontext_revise()
    model = model.to(device)

    diffusion = create_diffusion(timestep_respacing="") # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL()
    try:
        vae.load_state_dict(torch.load('weight_lol.pth')['vae'], strict=True)
        print('loading pretrained vae')
    except:
        print('error')
    vae = vae.to(device)

    cond_lq = CondEncoder()
    cond_lq = cond_lq.to(device)

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    parmas = list(model.parameters()) +  list(cond_lq.parameters())
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper)
    optimizer = torch.optim.AdamW(parmas, lr=1e-4, weight_decay=0)
    scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2500, 3500, 4000, 4500, 4800], gamma=0.5)
    
    dataset_cls = LoL_Dataset_RIDCP

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = dataset_cls(opt=dataset_opt, train=True)
            print('complete dataset')
            train_loader = create_dataloader(True, train_set, dataset_opt, opt, None)
            
        elif phase == 'val':
            val_set = dataset_cls(opt=dataset_opt, train=False)
            val_loader = create_dataloader(False, val_set, dataset_opt, opt, None)
    
    print('complete trainloader')
    
    model.train() 

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch+1}...")
        scheduler_G.step()
        for data in train_loader:
            x = data['GT'].to(device)
            y = data['LQ'].to(device)
            global_prior = data['global'].to(device)
            local_prior = data['local'].to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).to(device) 
                
            y = cond_lq(y).to(device)
            
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y, vis=global_prior, q_map=local_prior)
            
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device='cuda')
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()
                        
        if (epoch +1) <= 2000 :
            val_fre = 100
        elif (epoch +1) <= 4000 and (epoch +1) > 2000 :
            val_fre = 50
        else:
            val_fre = 10

        if (epoch + 1) % val_fre == 0 or epoch == 0:
            model.eval()
            img_dir = os.path.join(opt['path']['val_images'], str(epoch+1))
            util.mkdirs(img_dir)
            diffusion_val = create_diffusion(str(args.num_sampling_steps))
            for data in val_loader:
                x = data['GT'].to(device)
                y = data['LQ'].to(device)
                global_prior = data['global'].to(device)
                local_prior = data['local'].to(device)
                padding_params = data['padding_params']
                b, c, h, w = y.shape
                with torch.no_grad():
                    y = cond_lq(y).to(device)
                latent_size_h = h // 4
                latent_size_w = w // 4
                z = torch.randn(1, 3, latent_size_h, latent_size_w, device=device)
                model_kwargs = dict(y=y, vis=global_prior, q_map=local_prior)

                # Sample images:
                samples = diffusion_val.p_sample_loop(
                    model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                    )
                with torch.no_grad():
                    samples = vae.decode(samples)
                samples = samples[:, :, padding_params[0]:samples.shape[2] - padding_params[1],padding_params[2]:samples.shape[3] - padding_params[3]]
                
                assert samples.shape == x.shape                     
                save_img_path = os.path.join(img_dir, os.path.basename(data['GT_path'][0]+'.png'))                   
                save_image(samples, save_img_path, range=(0, 1))
            model.train()
        
        if (epoch +1) <= 2000 :
            save_fre = 100
        elif (epoch +1) <= 4000 and (epoch +1) > 2000 :
            save_fre  = 50
        else:
            save_fre  = 10

        # Save DiT checkpoint:
        if (epoch +1) % save_fre == 0 or epoch ==0 :
            if rank <=0:
                util.mkdirs(os.path.join(opt['path']['models'], str(epoch+1)))
                save_filename = '{}.pth'.format(epoch+1)
                save_path = os.path.join(opt['path']['models'], str(epoch+1), save_filename)
                save_filename_ll = '{}_condencoder.pth'.format(epoch+1)
                save_path_ll = os.path.join(opt['path']['models'], str(epoch+1), save_filename_ll)
                    
                torch.save(model.state_dict(), save_path)
                torch.save(cond_lq.state_dict(), save_path_ll)
                    
                logger.info(f"Saved checkpoint to {save_path}")

    model.eval()  
    logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option YMAL file.',
                            default='LOLv1_dit.yml')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=5000)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--num-sampling-steps", type=int, default=25)
    parser.add_argument("--train", type=bool, default=True)
    args = parser.parse_args()
    main(args)
