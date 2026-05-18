<div align="center">

<h2>T2I-Diff: fMRI Signal Generation via Time-Frequency Image Transform and Classifier-Free Denoising Diffusion Models</h2>

**_The first framework to introduce time-frequency image transform for fMRI BOLD signal generation and brain disorder classification!_**

[Hwa Hui Tew](https://htew0001.github.io/)¹, [Junn Yong Loo](https://scholar.google.com/citations?user=PsL3CMYAAAAJ&hl=en)¹*, [Chee-Ming Ting](https://scholar.google.com.my/citations?user=z_BmSq4AAAAJ&hl=en&authuser=1&oi=ao)¹*

¹ ​School of Information Technology, Monash University Malaysia

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reconstruction-vs-generation-taming-1/image-generation-on-imagenet-256x256)](https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?p=reconstruction-vs-generation-taming-1)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-VA_VAE-b31b1b.svg)]()
[![arXiv](https://img.shields.io/badge/arXiv-FasterDiT-b31b1b.svg)](https://arxiv.org/abs/2410.10356) -->
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![authors](https://img.shields.io/badge/by-hustvl-green)](https://github.com/hustvl)
[![paper](https://img.shields.io/badge/CVPR'25-VA_VAE-b31b1b.svg)](https://arxiv.org/abs/2501.01423)
[![arXiv](https://img.shields.io/badge/NeurIPS'24-FasterDiT-b31b1b.svg)](https://arxiv.org/abs/2410.10356)


</div>
<div align="center">
<img src="images/vis.png" alt="Visualization">
</div>

## 📰 News

- **[2026.01.26]** **Check our new work [DSFM](https://openreview.net/pdf?id=Dgphd9qizu), a new generative modeling framework for fMRI !**

- **[2025.05.18]** We have released the code and paper for T2I-Diff !

## 📄 Abstract

Functional Magnetic Resonance Imaging (fMRI) is an advanced neuroimaging method that enables in-depth analysis of brain activity by measuring dynamic changes in the blood oxygenation leveldependent (BOLD) signals. However, the resource-intensive nature of fMRI data acquisition limits the availability of high-fidelity samples required for data-driven brain analysis models. While modern generative models can synthesize fMRI data, they often underperform because they overlook the complex non-stationarity and nonlinear BOLD dynamics. 

To address these challenges, we introduce T2I-Diff, an fMRI generation framework that leverages time-frequency representation of BOLD signals and classifier-free denoising diffusion. Specifically, our framework first converts BOLD signals into windowed spectrograms via a timedependent Fourier transform, capturing both the underlying temporal dynamics and spectral evolution. Subsequently, a classifier-free diffusion model is trained to generate class-conditioned frequency spectrograms, which are then reverted to BOLD signals via inverse Fourier transforms. Finally, we validate the efficacy of our approach by demonstrating improved accuracy and generalization in downstream fMRI-based brain network classification. 

## 🎯 How to Use

### Installation

```
conda create -n lightningdit python=3.10.12
conda activate lightningdit
pip install -r requirements.txt
```


### Inference with Pre-trained Models

- Download weights and data infos:

    - Download pre-trained models
        | Tokenizer | Generation Model | FID | FID cfg |
        |:---------:|:----------------|:----:|:---:|
        | [VA-VAE](https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/blob/main/vavae-imagenet256-f16d32-dinov2.pt) | [LightningDiT-XL-800ep](https://huggingface.co/hustvl/lightningdit-xl-imagenet256-800ep/blob/main/lightningdit-xl-imagenet256-800ep.pt) | 2.17 | 1.35 |
        |           | [LightningDiT-XL-64ep](https://huggingface.co/hustvl/lightningdit-xl-imagenet256-64ep/blob/main/lightningdit-xl-imagenet256-64ep.pt) | 5.14 | 2.11 |

    - Download [latent statistics](https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/blob/main/latents_stats.pt). This file contains the channel-wise mean and standard deviation statistics.

    - Modify config file in ``configs/reproductions`` as required. 

- Fast sample demo images:

    Run:
    ```
    bash bash run_fast_inference.sh ${config_path}
    ```
    Images will be saved into ``demo_images/demo_samples.png``, e.g. the following one:
    <div align="center">
    <img src="images/demo_samples.png" alt="Demo Samples" width="600">
    </div>

- Sample for FID-50k evaluation:
    
    Run:
    ```
    bash run_inference.sh ${config_path}
    ```
    NOTE: The FID result reported by the script serves as a reference value. The final FID-50k reported in paper is evaluated with ADM:

    ```
    git clone https://github.com/openai/guided-diffusion.git
    
    # save your npz file with tools/save_npz.py
    bash run_fid_eval.sh /path/to/your.npz
    ```

## ❤️ Acknowledgements

This repo is mainly built on [ImagenTime](https://github.com/azencot-group/ImagenTime/tree/main). Thanks for the great work.

## 📝 Citation

If you find our work useful, please cite our related paper:

```
# ICLR 2026
@inproceedings{tewfunctional,
  title={Functional MRI Time Series Generation via Wavelet-Based Image Transform and Spectral Flow Matching for Brain Disorder Identification},
  author={Tew, Hwa Hui and Loo, Junn Yong and Yu, Leong Fang and Lau, Julia K and Fan, Ding and Ombao, Hernando and Phan, Raphael CW and Tan, Chee Pin and Ting, Chee-Ming},
  booktitle={The Fourteenth International Conference on Learning Representations}
}

# MICCAI 2025
@inproceedings{tew2025t2i,
  title={T2I-Diff: fMRI Signal Generation via Time-Frequency Image Transform and Classifier-Free Denoising Diffusion Models},
  author={Tew, Hwa Hui and Loo, Junn Yong and Tan, Yee-Fan and Tang, Xinyu and Ombao, Hernando and Noman, Fuad and Phan, Rapha{\"e}l C-W and Ting, Chee-Ming},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={640--650},
  year={2025},
  organization={Springer}
}
```
