<div align="center">

<h2>T2I-Diff: fMRI Signal Generation via Time-Frequency Image Transform and Classifier-Free Denoising Diffusion Models</h2>

**_The first framework to introduce time-frequency image transform for fMRI BOLD signal generation and brain disorder classification!_**

[Hwa Hui Tew](https://htew0001.github.io/)¹, [Junn Yong Loo](https://scholar.google.com/citations?user=PsL3CMYAAAAJ&hl=en)¹, [Chee-Ming Ting](https://scholar.google.com.my/citations?user=z_BmSq4AAAAJ&hl=en&authuser=1&oi=ao)¹*

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

## ✨ Highlights

- Latent diffusion system with 0.28 rFID and **1.35 FID on ImageNet-256** generation!

- **More than 21.8× faster** convergence with **VA-VAE** and **LightningDiT** than original DiT!

- **Surpass DiT with FID=2.11 with only 8 GPUs in about 10 hours**. Let's make diffusion transformers research more affordable!

## 📰 News

- **[2026.01.26]** **Check our new work [DSFM](https://openreview.net/pdf?id=Dgphd9qizu) !**

- **[2025.05.18]** We have released the code and paper for T2I-Diff !

## 📄 Abstract

Functional Magnetic Resonance Imaging (fMRI) is an advanced neuroimaging method that enables in-depth analysis of brain activity by measuring dynamic changes in the blood oxygenation leveldependent (BOLD) signals. However, the resource-intensive nature of fMRI data acquisition limits the availability of high-fidelity samples required for data-driven brain analysis models. While modern generative models can synthesize fMRI data, they often underperform because they overlook the complex non-stationarity and nonlinear BOLD dynamics. 

To address these challenges, we introduce T2I-Diff, an fMRI generation framework that leverages time-frequency representation of BOLD signals and classifier-free denoising diffusion. Specifically, our framework first converts BOLD signals into windowed spectrograms via a timedependent Fourier transform, capturing both the underlying temporal dynamics and spectral evolution. Subsequently, a classifier-free diffusion model is trained to generate class-conditioned frequency spectrograms, which are then reverted to BOLD signals via inverse Fourier transforms. Finally, we validate the efficacy of our approach by demonstrating improved accuracy and generalization in downstream fMRI-based brain network classification. 

## 📝 Results

- State-of-the-art Performance on ImageNet 256x256 with FID=1.35.
- Surpass DiT within only 64 epochs training, achieving 21.8x speedup.

<div align="center">
<img src="images/results.png" alt="Results">
</div>

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

## 🎮 Train Your Own Models

 
- **We provide a 👆[detailed tutorial](docs/tutorial.md) for training your own models of 2.1 FID score within only 64 epochs. It takes only about 10 hours with 8 x H800 GPUs.** 


## ❤️ Acknowledgements

This repo is mainly built on [DiT](https://github.com/facebookresearch/DiT), [FastDiT](https://github.com/chuanyangjin/fast-DiT) and [SiT](https://github.com/willisma/SiT). Our VAVAE codes are mainly built with [LDM](https://github.com/CompVis/latent-diffusion) and [MAR](https://github.com/LTH14/mar). Thanks for all these great works.

## 📝 Citation

If you find our work useful, please cite our related paper:

```
# CVPR 2025
@inproceedings{yao2025vavae,
  title={Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models},
  author={Yao, Jingfeng and Yang, Bin and Wang, Xinggang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}

# NeurIPS 2024
@article{yao2024fasterdit,
  title={Fasterdit: Towards faster diffusion transformers training without architecture modification},
  author={Yao, Jingfeng and Wang, Cheng and Liu, Wenyu and Wang, Xinggang},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={56166--56189},
  year={2024}
}
```
