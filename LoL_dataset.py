import os
import subprocess
import torch.utils.data as data
import numpy as np
import time
import torch
import pickle
import cv2
from torchvision.transforms import ToTensor, Normalize
import random
import torchvision.transforms as T
from torchvision.utils import make_grid
import math

def rgb(t): 
    return (
        np.clip((t[0] if len(t.shape) == 4 else t).numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)
    

def auto_padding(img, times=8):
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

# import pdb
def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise IOError('Failed in writing images.')
    
def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = np.int64(img_np).astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

def random_resize(img, scale_factor=1.):
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

def resize(img, gt_size=256):
    return cv2.resize(img, (gt_size, gt_size), interpolation=cv2.INTER_CUBIC)

def paired_random_crop(img_gts, img_lqs, quality_map, gt_patch_size, scale):

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    if not isinstance(quality_map, list):
        quality_map = [quality_map]

        

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
        h_map, w_map = quality_map[0].size()[-2:]
        
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
        h_map, w_map = quality_map[0].shape[0:2]
        
    lq_patch_size = gt_patch_size // scale
    map_patch_size =gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
        
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]
        

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]


    # crop corresponding gt patch
    top_map, left_map = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        quality_map = [v[:, :, top_map:top_map + gt_patch_size, left_map:left_map + gt_patch_size] for v in quality_map]
    else:
        quality_map = [v[top_map:top_map + gt_patch_size, left_map:left_map + gt_patch_size, ...] for v in quality_map]
    
    
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]

    if len(quality_map) == 1:
        quality_map = quality_map[0]
        
    
    return img_gts, img_lqs, quality_map


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


class LoL_Dataset_RIDCP(data.Dataset): 
    def __init__(self, opt, train):
        self.opt = opt
        self.root = opt["root"]
        self.crop_size = self.opt['GT_size']
        
        if train:
            self.root = os.path.join(self.root, 'all')
        else:
            self.root = os.path.join(self.root, 'test')
        if train:
            self.pairs = self.load_pairs(self.root)
        else:
            self.pairs = self.load_pairs_val(self.root)
        
        self.to_tensor = ToTensor()

        

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        low_list = os.listdir(os.path.join(folder_path, 'low'))
        vis_list = os.listdir(os.path.join(folder_path, 'global_score'))
        quality_map_list = os.listdir(os.path.join(folder_path, 'local_prior'))
        print(len(vis_list))

        pairs = []
        for idx, f_name in enumerate(zip(low_list, vis_list, quality_map_list)):
            pairs.append(
                [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name[0])), cv2.COLOR_BGR2RGB),
                 cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name[0])), cv2.COLOR_BGR2RGB),
                 f_name[0].split('.')[0],
                 torch.load(os.path.join(folder_path, 'local_prior', f_name[2]), map_location='cpu'),#cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'local_prior', f_name[2])), cv2.COLOR_BGR2RGB),
                 torch.load(os.path.join(folder_path, 'global_score', f_name[1]),  map_location='cpu')])
        return pairs
    

    def load_pairs_val(self, folder_path):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        low_list = os.listdir(os.path.join(folder_path, 'low'))
        print(len(low_list))
        vis_list = os.listdir(os.path.join(folder_path, 'global_score'))
        quality_map_list = os.listdir(os.path.join(folder_path, 'local_prior'))
        pairs = []
        for idx, f_name in enumerate(zip(low_list, vis_list, quality_map_list)):
            lr, padding_params = auto_padding(cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name[0])), cv2.COLOR_BGR2RGB))
            #quality_map, padding_params = auto_padding(cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'local_prior', f_name[2])), cv2.COLOR_BGR2RGB))
        
            pairs.append(
                [lr,
                 cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name[0])), cv2.COLOR_BGR2RGB),
                 f_name[0].split('.')[0],
                 torch.load(os.path.join(folder_path, 'local_prior', f_name[2]), map_location='cpu'),
                 torch.load(os.path.join(folder_path, 'global_score', f_name[1]), map_location='cpu'),
                 padding_params])
        return pairs


    def __getitem__(self, item):
        
        if self.opt['phase'] == 'train':
            lr, hr, f_name, quality_map, vis = self.pairs[item]
            input_gt_size = np.min(hr.shape[:2])
            input_lq_size = np.min(lr.shape[:2])
            scale = input_gt_size // input_lq_size

            
            if self.opt['use_resize_crop']:
                
                # random resize
                input_gt_random_size = random.randint(self.crop_size, input_gt_size)
                input_gt_random_size = input_gt_random_size - input_gt_random_size % scale
                resize_factor = input_gt_random_size / input_gt_size

                hr = random_resize(hr, resize_factor)
                lr= random_resize(lr, resize_factor)
                

                quality_map = rgb(quality_map)

                quality_map = random_resize(quality_map, resize_factor)

                
                hr, lr, quality_map = paired_random_crop(hr, lr, quality_map, self.crop_size, input_gt_size // input_lq_size)
            
            #hr, lr, quality_map = augment([hr, lr, quality_map], self.opt['use_flip'],self.opt['use_rot'])
            
            hr = self.to_tensor(hr)
            lr = self.to_tensor(lr)

            quality_map = self.to_tensor(quality_map)

            #print(quality_map.shape)

            return {'LQ': lr, 'GT': hr, 'LQ_path': f_name, 'GT_path': f_name, 'global':vis, 'local': quality_map}
            
        if self.opt['phase'] == 'val':
            lr, hr, f_name, quality_map, vis, padding_params = self.pairs[item]
            
            hr = self.to_tensor(hr)
            lr = self.to_tensor(lr)
            
            #quality_map = self.to_tensor(quality_map)
            #print(quality_map.shape)
            return {'LQ': lr, 'GT': hr,  'GT_path': f_name, 'padding_params': padding_params, 'global':vis, 'local': quality_map.squeeze(0)}
        


def random_flip(img, seg, his_eq):
    random_choice = np.random.choice([True, False])
    img = img if random_choice else np.flip(img, 1).copy()
    seg = seg if random_choice else np.flip(seg, 1).copy()
    if his_eq is not None:
        his_eq = his_eq if random_choice else np.flip(his_eq, 1).copy()
    return img, seg, his_eq


def gamma_aug(img, gamma=0):
    max_val = img.max()
    img_after_norm = img / max_val
    img_after_norm = np.power(img_after_norm, gamma)
    return img_after_norm * max_val


def random_rotation(img, seg, his):
    random_choice = np.random.choice([0, 1, 3])
    img = np.rot90(img, random_choice, axes=(0, 1)).copy()
    seg = np.rot90(seg, random_choice, axes=(0, 1)).copy()
    if his is not None:
        his = np.rot90(his, random_choice, axes=(0, 1)).copy()
    return img, seg, his


def random_crop(hr, lr, his_eq, size_hr):
    size_lr = size_hr

    size_lr_x = lr.shape[0]
    size_lr_y = lr.shape[1]

    start_x_lr = np.random.randint(low=0, high=(size_lr_x - size_lr) + 1) if size_lr_x > size_lr else 0
    start_y_lr = np.random.randint(low=0, high=(size_lr_y - size_lr) + 1) if size_lr_y > size_lr else 0

    # LR Patch
    lr_patch = lr[start_x_lr:start_x_lr + size_lr, start_y_lr:start_y_lr + size_lr, :]

    # HR Patch
    start_x_hr = start_x_lr
    start_y_hr = start_y_lr
    hr_patch = hr[start_x_hr:start_x_hr + size_hr, start_y_hr:start_y_hr + size_hr, :]

    # HisEq Patch
    his_eq_patch = None
    if his_eq is not None:
        his_eq_patch = his_eq[start_x_hr:start_x_hr + size_hr, start_y_hr:start_y_hr + size_hr, :]
    return hr_patch, lr_patch, his_eq_patch


def center_crop(img, size):
    if img is None:
        return None
    assert img.shape[1] == img.shape[2], img.shape
    border_double = img.shape[1] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[border:-border, border:-border, :]


def center_crop_tensor(img, size):
    assert img.shape[2] == img.shape[3], img.shape
    border_double = img.shape[2] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[:, :, border:-border, border:-border]




'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data


def create_dataloader(train, dataset, dataset_opt, opt=None, sampler=None):
    gpu_ids = opt.get('gpu_ids', None)
    #gpu_ids = gpu_ids if gpu_ids else []
    num_workers = dataset_opt['n_workers']* (len(gpu_ids)+1)
    batch_size = dataset_opt['batch_size']
    shuffle = True
    if train:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                           num_workers=num_workers, sampler=sampler, drop_last=False,
                                           pin_memory=False)


# def create_dataset(dataset_opt):
#     print(dataset_opt)
#     mode = dataset_opt['mode']
#     if mode == 'LoL':
#         from data.LoL_dataset import LoL_Dataset as D
#     else:
#         raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
#     dataset = D(dataset_opt)

#     logger = logging.getLogger('base')
#     logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
#                                                            dataset_opt['name']))
#     return dataset