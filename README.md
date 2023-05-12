# PyTorch_ImageClassifiction_Techniques

For MSDM5055: A survey on deep learning image classification technics based on PyTorch and PyTorch lightning

- [PyTorch\_ImageClassifiction\_Techniques](#pytorch_imageclassifiction_techniques)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Usage](#usage)
  - [Reference](#reference)


## Introduction
This is a repository for the final project of MSDM5055 about applying image classification techniques in CIFAR-10 dataset with PyTorch lightning.


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
- [ ] Data Augmentation: just use transforms
- [ ] Captum: visualization and model explanation



## Usage
1. Download Google Drive Desktop. [Google Drive](https://www.google.com/drive/download/)
2. Create a folder named '5005' in Google Drive Desktop (drive/MyDrive/5055/). 
3. Clone this repository to the folder.
4. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucaJiang/PyTorch_ImageClassifiction_Techniques/blob/main/runcodes.ipynb)Open runcodes.ipynb in Colab, which will guide you to download the dependencies and mount the Google Drive.





## Reference
Write down all your references here. 
1. PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/
2. PyTorch Lightning CIFAR-10: https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html
3. Training tricks: https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
4. Graph Attention: https://www.baeldung.com/cs/graph-attention-networks
5. Pytorch Lightning models with Weights & Biases: https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Optimize_Pytorch_Lightning_models_with_Weights_%26_Biases.ipynb
6. 