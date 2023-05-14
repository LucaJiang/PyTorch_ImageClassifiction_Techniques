# PyTorch_ImageClassifiction_Techniques

For MSDM5055: A survey on deep learning image classification technics based on PyTorch and PyTorch lightning

- [PyTorch\_ImageClassifiction\_Techniques](#pytorch_imageclassifiction_techniques)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Usage](#usage)
  - [Run in Colab](#run-in-colab)
  - [Reference](#reference)




## Introduction
This is a repository for the final project of MSDM5055 about applying image classification techniques in CIFAR-10 dataset with PyTorch.


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
- [ ] Data Augmentation: wait uncomment



## Usage

## Run in Colab
1. Download Google Drive Desktop. [Google Drive](https://www.google.com/drive/download/)
2. Create a folder in Google Drive Desktop. 
3. Clone this repository to the folder.
4. Open runcodes.ipynb in Colab, which will guide you to download the dependencies and mount the Google Drive.


<!-- todo: directly open notebook in colab -->
5. Open the notebook in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucaJiang/PyTorch_ImageClassifiction_Techniques/blob/main/PyTorch_ImageClassifiction_Techniques.ipynb)
6. 

https://medium.com/analytics-vidhya/how-to-use-google-colab-with-github-via-google-drive-68efb23a42d






## Reference
Write down all your references here. 
1. PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/
2. PyTorch Lightning CIFAR-10: https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html
3. Training tricks: https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
4. Graph Attention: https://www.baeldung.com/cs/graph-attention-networks