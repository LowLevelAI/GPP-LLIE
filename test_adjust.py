import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import glob
import os
from model_incontext_revise import DiT_incontext_revise
from diffusion import create_diffusion
from vae.autoencoder import AutoencoderKL
from vae.cond_encoder import CondEncoder
from vae.encoder_decoder import Decoder2
from utils import util
from torchvision.utils import save_image
from download import load_model
from torch.nn import functional as F
import natsort

import cv2
import numpy as np

def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))
def t(array): return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255
def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)
def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def main(inp_dir):

    lr_dir = os.path.join(inp_dir, 'low')
    high_dir = os.path.join(inp_dir, 'high')
    global_prior_dir = os.path.join(inp_dir, 'global_score')
    local_prior_dir = os.path.join(inp_dir, 'local_prior')

    out_dir = os.path.join(inp_dir, 'outputs_adjust')
    os.makedirs(out_dir, exist_ok=True)

    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.png'))
    hr_paths = fiFindByWildcard(os.path.join(high_dir, '*.png'))
    global_prior_paths = fiFindByWildcard(os.path.join(global_prior_dir, '*.pt'))
    local_prior_paths = fiFindByWildcard(os.path.join(local_prior_dir, '*.pt'))

    device = torch.device('cuda:0')
    state_dict = torch.load('weight_lol.pth')

    model = DiT_incontext_revise()
    model.load_state_dict(state_dict['dit'], strict=True)
    model = model.to(device)

    vae = AutoencoderKL()
    vae.load_state_dict(state_dict['vae'], strict=True)
    vae = vae.to(device)

    cond_lq = CondEncoder()
    cond_lq.load_state_dict(state_dict['cond'], strict=True)
    cond_lq = cond_lq.to(device)

    second_decoder = Decoder2()
    second_decoder.load_state_dict(state_dict['second_decoder'], strict=True)
    second_decoder = second_decoder.to(device)

    model.eval()
    diffusion_val = create_diffusion(str(25))  # number of sample steps 

    for lr_path, hr_path, global_path, local_path, test_index in zip(lr_paths, hr_paths, global_prior_paths, local_prior_paths, range(len(lr_paths))):
        
        y = t(imread(lr_path)).to(device)
        hr = imread(hr_path)

        global_prior = torch.load(global_path).to(device)
        local_prior = torch.load(local_path).to(device)

        b, c, h, w = y.shape
        with torch.no_grad():
            y, enc_feat = cond_lq(y.to(device), True)
            latent_size_h = h // 4
            latent_size_w = w // 4
            z = torch.randn(1, 3, latent_size_h, latent_size_w, device=device)
            model_kwargs = dict(y=y, vis=global_prior, q_map=local_prior)

            # Sample images:
            samples = diffusion_val.p_sample_loop(
                model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                )

            dec_feat = vae.decode(samples, mid_feat=True)
            
            sr = second_decoder(samples, dec_feat, enc_feat)

        mean_sr = sr.reshape(sr.shape[0],-1).mean(dim=1)
        mean_hr = cv2.cvtColor(hr.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()/255
        sr = torch.clamp(sr * ( mean_hr / mean_sr), 0, 1)  

        save_img_path = os.path.join(out_dir, os.path.basename(lr_path))                   
        save_image(sr, save_img_path, range=(0, 1))

    


if __name__ == "__main__":

    input_dir = 'dataset/LOLv1/Test'# update the input dir, which at least contains such sub-folder: low, high, global_score, local_prior

    main(input_dir)
