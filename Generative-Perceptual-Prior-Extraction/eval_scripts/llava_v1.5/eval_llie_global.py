import argparse
import torch

import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from tqdm import tqdm

import requests
from PIL import Image
from io import BytesIO

from scipy.stats import spearmanr, pearsonr
from scipy.stats import kendalltau


import numpy as np
import glob
import natsort
import cv2

def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def softmax(a, b, temperature=100):
    a /= temperature
    b /= temperature
    return np.exp(a) / (np.exp(a) + np.exp(b))

def my_softmax(a, b):
    return 1 / (1 + np.exp((b-a)/3))


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

#os.makedirs("results/mix-llava-v1.5-7b/", exist_ok=True)


def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)



def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])

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

def patchify_numpy(img, patchify_factor=32):
    h, w, c = img.shape

    img_padding, padding_params = auto_padding(img, patchify_factor)
    h1, w1, c = img_padding.shape

    assert h1 % patchify_factor == 0
    assert w1 % patchify_factor == 0

    patch_h = h1 / patchify_factor
    patch_w = w1 / patchify_factor

    output = []

    for i in range(patchify_factor):
        for j in range(patchify_factor):
            a = img[int(patch_h*(i)):int(patch_h*(i+1)), int(patch_w*(j)):int(patch_w*(j+1)), :]
            #print(a.shape)
            output.append(a)
    return output, padding_params
    

def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, True)

    qs = args.query
    qs1 = args.query1
    qs2 = args.query2

    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        qs1 = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs1
        qs2 = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs2
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        qs1 = DEFAULT_IMAGE_TOKEN + '\n' + qs1
        qs2 = DEFAULT_IMAGE_TOKEN + '\n' + qs2

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
 
    conv1 = conv_templates[args.conv_mode].copy()
    conv1.append_message(conv1.roles[0], qs1)
    conv1.append_message(conv1.roles[1], None)
    prompt1 = conv1.get_prompt()

    conv2 = conv_templates[args.conv_mode].copy()
    conv2.append_message(conv2.roles[0], qs2)
    conv2.append_message(conv2.roles[1], None)
    prompt2 = conv2.get_prompt()

    test_root = args.test_root 
    input_dir = test_root + '/low'
    image_paths = fiFindByWildcard(os.path.join(input_dir, '*.*'))
    print(len(image_paths))

    out_dir = test_root + '/global_score'
    os.makedirs(out_dir, exist_ok=True)

    for image_path, _ in zip(image_paths, range(len(image_paths))):
        image = load_image(image_path)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        
      
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
            output_logits = model(input_ids, images=image_tensor)["logits"][:,-1]
        good, poor = output_logits[0,1781].item(), output_logits[0,6460].item() 
        score_vis = my_softmax(good, poor)


        input_ids1 = tokenizer_image_token(prompt1, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str1 = conv1.sep if conv1.sep_style != SeparatorStyle.TWO else conv1.sep2
        keywords1 = [stop_str1]
        stopping_criteria = KeywordsStoppingCriteria(keywords1, tokenizer, input_ids1)
        with torch.inference_mode():
            output_logits1 = model(input_ids1, images=image_tensor)["logits"][:,-1]
        good1, poor1 = output_logits1[0,1781].item(), output_logits1[0,6460].item()
        score_contrast = my_softmax(good1, poor1)


        input_ids2 = tokenizer_image_token(prompt2, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str2 = conv2.sep if conv2.sep_style != SeparatorStyle.TWO else conv2.sep2
        keywords2 = [stop_str2]
        stopping_criteria = KeywordsStoppingCriteria(keywords2, tokenizer, input_ids2)
        with torch.inference_mode():
            output_logits2 = model(input_ids2, images=image_tensor)["logits"][:,-1]
        good2, poor2 = output_logits2[0,1781].item(), output_logits2[0,6460].item()
        score_sharpness = my_softmax(good2, poor2)


        score = (score_vis + score_contrast + score_sharpness) / 3
        score_t = torch.tensor(score)
        save_path = os.path.join(out_dir, os.path.basename(image_path).split('.')[0]+'.pt')
        torch.save(score_t, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="teowu/llava_v1.5_7b_qinstruct_preview_v0.1")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--query", type=str, default="Visibility refers to the clarity and distinctness with which details of an image can be seen and recognized. Rate the visibility of the image.")
    parser.add_argument("--query1", type=str, default="Contrast refers to the difference in luminance or color that makes an object distinguishable from others within an image. Rate the contrast of the image.")
    parser.add_argument("--query2", type=str, default="Sharpness refers to the clarity of detail and the edge definition in an image. Rate the sharpness of the image.")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--test-root", type=str, default="dataset/LOLv2-syn/Train")
    args = parser.parse_args()

    eval_model(args)
