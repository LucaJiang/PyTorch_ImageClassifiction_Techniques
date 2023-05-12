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
- [ ] 


## Reference
Write down all your references here. 
1. PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/
2. PyTorch Lightning CIFAR-10: https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html
3. Training tricks: https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
4. Graph Attention: https://www.baeldung.com/cs/graph-attention-networks