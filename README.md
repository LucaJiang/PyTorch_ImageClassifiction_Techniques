# PyTorch_ImageClassifiction_Techniques

For MSDM5055: A survey on deep learning image classification technics based on PyTorch and PyTorch lightning

- [PyTorch\_ImageClassifiction\_Techniques](#pytorch_imageclassifiction_techniques)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Run in Colab](#run-in-colab)
  - [Reference](#reference)


## Introduction
This is a repository for the final project of MSDM5055 about applying image classification techniques in CIFAR-10 dataset with PyTorch.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucaJiang/PyTorch_ImageClassifiction_Techniques/blob/main/runcodes.ipynb) 

## Features
- [x] Graph Attention Block
- [x] PyTorch Lightning
  - [x] Effective Training Techniques
    - [x] Accumulate Gradients: GradientAccumulationScheduler
    - [x] Gradient Clipping: gradient_clip_val
    - [x] Stochastic Weight Averaging: StochasticWeightAveraging
    - [x] Batch Size Finder: tuner.scale_batch_size(model, mode="power")
    - [x] Learning Rate Finder: tuner.lr_find(model)
  - [ ] Fine-tune Scheduler: debug
- [x] Data Augmentation: Cifar-10 generation with conditional diffusion model


## Run in Colab
1. Download [Google Drive Desktop](https://www.google.com/drive/download/).
2. Create a folder in your Google Drive Desktop. 
3. Clone this repository to the folder.
4. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucaJiang/PyTorch_ImageClassifiction_Techniques/blob/main/runcodes.ipynb)Open runcodes.ipynb in Colab, which will guide you to download the dependencies and mount the Google Drive.


## Reference
Write down all your references here. 
1. PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/
2. PyTorch Lightning CIFAR-10: https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html
3. Training tricks: https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
4. Graph Attention: https://www.baeldung.com/cs/graph-attention-networks
5. Diffusion Model:https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#:~:text=Diffusion%20models%20are%20inspired%20by,data%20samples%20from%20the%20noise
6. Cifar-10 Generation with Diffusion Model:https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-
