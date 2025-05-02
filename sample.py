# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from vae.autoencoder import AutoencoderKL
from download import find_model
from model_incontext_revise import DiT_incontext_revise
import argparse
import options.options as option
from LoL_dataset import LoL_Dataset_RIDCP, create_dataloader
from utils import util
import natsort
import pandas as pd
import os
from Measure import Measure, psnr
import numpy as np
import cv2
import glob
from collections import OrderedDict
from vae.autoencoder import AutoencoderKL
from vae.cond_encoder import CondEncoder


from vae.encoder_decoder import Decoder2

def auto_padding2(img, times=8):
    # img: numpy image with shape H*W*C

    h, w, _ = img.shape

    if h % times == 0:
        h1 = 0
        h2 = 0 
    else: 
        h1 = (times - h % times) // 2
        h2 = (times - h % times) - h1

    if w % times == 0:
        w1 = 0
        w2 = 0 
    else: 
        w1 = (times - w % times) // 2
        w2 = (times - w % times) - w1
        
    img = cv2.copyMakeBorder(img, h1, h2, w1, w2, cv2.BORDER_REFLECT)
    return img, [h1, h2, w1, w2]


def auto_padding(img, times=16):
    # img: numpy image with shape H*W*C

    h, w, _ = img.shape
    h1, w1 = (times - h % times) // 2, (times - w % times) // 2
    h2, w2 = (times - h % times) - h1, (times - w % times) - w1
    img = cv2.copyMakeBorder(img, h1, h2, w1, w2, cv2.BORDER_REFLECT)
    return img, [h1, h2, w1, w2]

def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))
def t(array): return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255

def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda:0"

    vae = AutoencoderKL()
    ckpt_vae = 'weights/vae.pkl'
    state_dict = find_model(ckpt_vae)
    try:
        vae.load_state_dict(state_dict, strict=True)
        print('loading pretrained vae')
    except:
        print('error')
    vae = vae.to(device)
    
    model = DiT_incontext_revise()
    ckpt_path = 'checkpoints/dit.pth'
    state_dict = find_model(ckpt_path)
    try:
        model.load_state_dict(state_dict, strict=True)
        print('loading pretrained GPP_LLIE model')
    except:
        print('error')
    model = model.to(device)
    model.eval() 

    cond_lq = CondEncoder()
    ckpt_condencoder = 'weights/condencoder.pth'
    state_dict = find_model(ckpt_condencoder)
    try:
        cond_lq.load_state_dict(state_dict, strict=True)
        print('loading pretrained cond_encoder')
    except:
        print('error')
    cond_lq = cond_lq.to(device)    
    cond_lq.eval()  
    
    defor_decoder = Decoder2()  ## dual-decoder, deformable modulation
    ckpt_defor =os.path.join('weights/defordecoder.pth')
    state_dict_defor = find_model(ckpt_defor)
    try:
        defor_decoder.load_state_dict(state_dict_defor, strict=True)
        print('loading defor decoder')
    except:
        print('loading defor decoder error')
    defor_decoder = defor_decoder.to(device)
    defor_decoder.eval()  # important!

    diffusion = create_diffusion(str(args.num_sampling_steps))
    

    lr_dir = 'LOLdataset/eval15/low'
    gt_dir = 'LOLdataset/eval15/high'
    vis_dir = 'LOLdataset//eval15/total_score'
    quality_dir = 'LOLdataset//eval15/quality_map'

    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.*'))
    gt_paths = fiFindByWildcard(os.path.join(gt_dir, '*.*'))
    vis_paths = fiFindByWildcard(os.path.join(vis_dir, '*.*'))
    quality_paths = fiFindByWildcard(os.path.join(quality_dir, '*.*'))

    test_dir = 'result_LOL'
    

      
    for lr_path, gt_path, vis_path, quality_path, idx_test in zip(lr_paths, gt_paths, vis_paths, quality_paths, range(len(lr_paths))):

        hr = imread(gt_path)
        hr_t = t(hr)
            
        lr = imread(lr_path)
        quality_map = imread(quality_path)
        vis_t =torch.load(vis_path).to(device)
        h0, w0, c = lr.shape


        lr, padding_params = auto_padding2(lr)
        lr_t = t(lr).to(device)
        quality_map, _ = auto_padding2(quality_map)
        quality_map_t = t(quality_map).to(device)
        h, w, c = lr.shape
        latent_size_h = h // 4
        latent_size_w = w // 4
        z = torch.randn(1, 3, latent_size_h, latent_size_w, device=device)
        with torch.no_grad():
            enc_feat, y = cond_lq(lr_t, mid_feat=True)
            y = y.to(device)

        z = torch.randn(1, 3, latent_size_h, latent_size_w, device=device)

        model_kwargs = dict(y=y, vis=vis_t, q_map=quality_map_t)

        # Sample images:
        samples = diffusion.p_sample_loop(
                    model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                    )
        
        with torch.no_grad():
            code_decoder_output = vae.decode(samples, mid_feat=True)
            samples = defor_decoder(samples, code_decoder_output, enc_feat)
        samples = samples[:, :, padding_params[0]:samples.shape[2] - padding_params[1],padding_params[2]:samples.shape[3] - padding_params[3]]

        mean_out = samples.reshape(samples.shape[0],-1).mean(dim=1)
        mean_gt = cv2.cvtColor(hr.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()/255
        
        samples = rgb(torch.clamp(samples*(mean_gt/mean_out), 0, 1))

        samples_numpy = rgb(samples)
        

        save_img_path = os.path.join(test_dir, os.path.basename(lr_path))      
        imwrite(save_img_path, samples_numpy)
            



def format_measurements(meas):
    s_out = []
    for k, v in meas.items():
        v = f"{v:0.4f}" if isinstance(v, float) else v
        s_out.append(f"{k}: {v}")
    str_out = ", ".join(s_out)
    return str_out

            



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option YMAL file.',
                            default='LOL.yml')
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default='1/5000.pth',
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
