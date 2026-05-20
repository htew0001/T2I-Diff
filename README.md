<div align="center">

<h2>T2I-Diff: fMRI Signal Generation via Time-Frequency Image Transform and Classifier-Free Denoising Diffusion Models</h2>

**_The first framework to introduce time-frequency image transform for fMRI BOLD signal generation and brain disorder classification!_**

[Hwa Hui Tew](https://htew0001.github.io/)¹, [Junn Yong Loo](https://scholar.google.com/citations?user=PsL3CMYAAAAJ&hl=en)¹*, [Chee-Ming Ting](https://scholar.google.com.my/citations?user=z_BmSq4AAAAJ&hl=en&authuser=1&oi=ao)¹*

¹ ​School of Information Technology, Monash University Malaysia

[![paper](https://img.shields.io/badge/MICCAI'25-T2IDiff-b31b1b)](https://papers.miccai.org/miccai-2025/paper/3042_paper.pdf)
[![arXiv](https://img.shields.io/badge/ICLR'26-DSFM-b31b1b)](https://openreview.net/pdf?id=Dgphd9qizu)


</div>
<div align="center">
<img src="Images/viz1.png" alt="Visualization">
    
**Figure 1:** Overview of our proposed T2I-Diff framework.
</div>

## 📰 News

- **[2026.01.26]** **Check our new work [DSFM](https://openreview.net/pdf?id=Dgphd9qizu), a new generative modeling framework for fMRI !**

- **[2025.05.18]** We have released the code and paper for T2I-Diff !

## 📄 Abstract

Functional Magnetic Resonance Imaging (fMRI) is an advanced neuroimaging method that enables in-depth analysis of brain activity by measuring dynamic changes in the blood oxygenation leveldependent (BOLD) signals. However, the resource-intensive nature of fMRI data acquisition limits the availability of high-fidelity samples required for data-driven brain analysis models. While modern generative models can synthesize fMRI data, they often underperform because they overlook the complex non-stationarity and nonlinear BOLD dynamics. 

To address these challenges, we introduce T2I-Diff, an fMRI generation framework that leverages time-frequency representation of BOLD signals and classifier-free denoising diffusion. Specifically, our framework first converts BOLD signals into windowed spectrograms via a timedependent Fourier transform, capturing both the underlying temporal dynamics and spectral evolution. Subsequently, a classifier-free diffusion model is trained to generate class-conditioned frequency spectrograms, which are then reverted to BOLD signals via inverse Fourier transforms. Finally, we validate the efficacy of our approach by demonstrating improved accuracy and generalization in downstream fMRI-based brain network classification. 

## 🎯 How to Use

### Datasets
#### 1. NetSim 
```
https://www.fmrib.ox.ac.uk/datasets/netsim/index.html
```
#### 2. MDD 
```
https://rfmri.org/REST-meta-MDD
```
After preprocessing, place the processed data in the project’s empty `/data/<desired_dataset>` folder.

### Installation
Download and set up the repository:
```
https://github.com/htew0001/T2I-Diff
cd T2I-Diff
```

We provide a [`requirements.yaml`](requirements.yaml) file to easily create a Conda environment configured to run the model:
```
conda env create -f requirements.yaml
conda activate T2IDIFF
```

### Usage
We include three main scripts to perform different tasks:
- **Conditional Generation**: [`run_training.py`](run_training.py) - Executes the training of conditional generative task for disease and healthy-control groups.
- **Conditional Sampling**: [`run_inference.py`](run_inference.py) - Executes the sampling of conditional generative task for disease and healthy-control groups.
- **Evaluation Metrics**: [`run_evaluation.py`](run_evaluation.py) - Executes the evaluation of various time-series metrics.

**For Training of Conditional Generative Models:**
```
python run_training.py --config ./configs/conditional/<desired_dataset>.yaml
```

**For Sampling of Conditional Generative Models:**
```
python run_inference.py --config ./configs/conditional/<desired_dataset>.yaml
```

**For Evaluation of Conditional Generative Models:**
```
python run_evaluation.py 
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

## 📧 Contact
For any inquiries, please contact at hwa.tew@monash.edu.

