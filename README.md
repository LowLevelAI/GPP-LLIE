#  [AAAI 2025] Low-Light Image Enhancement via Generative Perceptual Priors [[Paper]](https://arxiv.org/pdf/2412.20916)

<h4 align="center">Han Zhou<sup>1,*</sup>, Wei Dong<sup>1,*</sup>, Xiaohong Liu<sup>2,&dagger;</sup>, Yunlun Zhang<sup>2</sup>, Guangtao Zhai<sup>2</sup>, Jun Chen<sup>1</sup></center>
<h4 align="center"><sup>1</sup>McMaster University, <sup>2</sup>Shanghai Jiao Tong University, 
<h4 align="center"><sup>*</sup>Equal Contribution, <sup>&dagger;</sup>Corresponding Author</center></center>



### Introduction
This repository represents the official implementation of our AAAI 2025 paper titled **Low-Light Image Enhancement via Generative Perceptual Priors**. If you find this repo useful, please give it a star ⭐ and consider citing our paper in your research. Thank you.

[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

We present **GPP-LLIE**, a novel **LLIE** framework with the guidance of **G**enerative **P**erceptual **P**riors.

- **VLM-based Generative Perceptual Priors Extraction Pipeline**: extract generative perceptual priors for LL images globally and locally based on pre-trained VLMs.
- **Transformer-based Diffusion Framework**: develop a Transformer-based diffusion framework for LLIE.
- **Guidance of Perceptual Priors in the Reverse Diffusion Process**: leverage global perceptual priors to modulate the
layer normalization (GPP-LN) and utilize local perceptual priors to guide the attention mechanism (LPP-Attn) to benefit the enhancement process.

### Our Proposed VLM-based Generative Perceptual Priors Extraction Pipeline 
![teaser](images/prior-extraction-pipeline.png)

### Overall Framework
![teaser](images/framework.jpg)

## 📢 News
**2025-6-12:** This repo has been updated. The proposed VLM-based generative perceptual priors extraction pipeline has been added. ⭐ <br>

## 🛠️ Setup

The  code was tested on:

- Python 3.8, CUDA 11.6, GeForce RTX 2080Ti or higher GPU RAM.

### 📦 Repository

Clone the repository (requires git):

```bash
git clone https://github.com/LowLevelAI/GPP-LLIE.git
cd GPP-LLIE
```

### 💻 Dependencies

- **Make Conda Environment:** 

    ```bash
    conda create -n gppllie python=3.8
    conda activate gppllie
    ```
- **Then install dependencies:**

  ```bash
  conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
  pip install pyiqa==0.1.4 pytorch-lightning==1.9.0 natsort  
  ```

- **Build CUDA extensions:**
  
  ```bash
  cd defor_cuda_ext
  BASICSR_EXT=True python setup.py develop
  ```

- **Move CUDA extensions** (/defor_cuda_ext/basicsr/ops/dcn/deform_conv_ext.xxxxxx.so) to the path: **/ops/dcn/**.


## ✏️ Contributing

Please refer to [this](CONTRIBUTING.md) instruction.

## 🎓 Citation

If you find this repo and our paper useful, please consider citing our paper:

```bibtex
@inproceedings{zhou2025gppllie,
  title={Low-light image enhancement via generative perceptual priors},
  author={Zhou, Han and Dong, Wei and Liu, Xiaohong and Zhang, Yulun and Zhai, Guangtao and Chen, Jun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={10},
  pages={10752--10760},
  year={2025}
}

```


## 🎫 License

This work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

By downloading and using the code and model you agree to the terms in the  [LICENSE](LICENSE.txt).

[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)
