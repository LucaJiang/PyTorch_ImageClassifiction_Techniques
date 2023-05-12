# PyTorch_ImageClassifiction_Techniques
For MSDM5055

## Introduction
This is a repository for the final project of MSDM5055 about applying image classification techniques in CIFAR-10 dataset with PyTorch.


## Features
- [x] PyTorch Lightning
  - [x] Effective Training Techniques
    - [x] Accumulate Gradients: GradientAccumulationScheduler
    - [x] Gradient Clipping: gradient_clip_val
    - [x] Stochastic Weight Averaging: StochasticWeightAveraging
    - [x] Batch Size Finder: tuner.scale_batch_size(model, mode="power")
    - [x] Learning Rate Finder: tuner.lr_find(model)
  - [ ] Fine-tune Scheduler 
- [ ] Data Augmentation



## Usage

## Run in Colab
https://medium.com/analytics-vidhya/how-to-use-google-colab-with-github-via-google-drive-68efb23a42d

to be continued...
1. Open the notebook in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucaJiang/PyTorch_ImageClassifiction_Techniques/blob/main/PyTorch_ImageClassifiction_Techniques.ipynb)
2. 







## Reference
Write down all your references here. 
1. PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/
1. PyTorch Lightning CIFAR-10: https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html
1. Training tricks: https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
1. 