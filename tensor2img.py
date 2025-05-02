import torch
import natsort
import pandas as pd
import os

import numpy as np
import cv2
import glob


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))

def t(array): return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255


def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)


def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])

dir = 'LOLdataset/our485/quality_map_tensor'
paths = fiFindByWildcard(os.path.join(dir, '*.*'))
print(len(paths))


for path, id in zip(paths, range(len(paths))):
    t = torch.load(path)
    #print(t.shape)

    t_img =rgb(t)
    print(t_img.shape)
    save_path = os.path.join('LOLdataset/our485/quality_map_img', os.path.basename(path).split('.')[0]+'.png')
    imwrite(save_path, t_img)