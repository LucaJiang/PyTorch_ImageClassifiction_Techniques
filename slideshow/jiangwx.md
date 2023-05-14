# Training and Tuning with Tricks for CIFAR-10 dataset in PyTorch and PyTorch Lightning

Jiang Wenxin

--------------------
<!-- ## PyTorch to PyTorch Lightning -->
![PyTorch to PyTorch Lightning](../img/pt_to_pl.png)

--------------------
```python
callbacks = [
    ModelCheckpoint(monitor="val_acc", mode="max"),
    LearningRateMonitor(logging_interval="step"),
    StochasticWeightAveraging(swa_lrs=1e-2),
    early_stopping,
]
trainer = Trainer(
    max_epochs=50,
    devices='auto',  
    logger=wandb_logger,
    callbacks=callbacks,
)
```
--------------------
##  Overview and Basics
What we need know about training a model:
* Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
* Model: ResNet18 or ResNet34 [TorchVision.Models](https://pytorch.org/vision/0.8/models.html)
* Loss Function: NLL(Negative Log-Likelihood)[^1]
* Optimizer: SGD(Stochastic Gradient Descent) or Adam(Adaptive Moment Estimation)[^2]
* Hyperparameters: Learning Rate, Batch Size, Schedule, etc.
[^1]: The same with CrossEntropyLoss in one-hot encoding.
[^2]: Adam is one of the most popular optimizers in deep learning.

--------------------
## Transforms: Data Augmentation
Tools: random crop, random flip, random rotation, etc.
Benefits of data augmentation:
* Increase the size of the dataset -> Reduce <mark>overfitting</mark>
* Improve generalization -> Improve the performance of the model
![data_aug](../img/data_augmentation.png)

--------------------
## Transforms: Data Normalization and Resizing
Tools: Normalize, Resize, etc.
Why data normalization?
* Easier to converge
* Prevent gradient explosion / vanish
* Make features have the same scale
  
Why data resizing?
* Reduce the size of the img -> Save time
* Fit the input size of your model[^3]
[^3]: But we choose to change the input size of our model.

--------------------
## Before Training:
### Tricks: Learning Rate Finder
Not to pick the lowest loss, but in the middle of the sharpest downward slope (red point).
![FindLR](../img/pl_lr_finder.png)
<!-- It determines a range of learning rates by gradually increasing the learning rate during training and observing the change in the loss function, thus helping us to better select the learning rate to improve the training effect and convergence speed of the model. -->

--------------------
## [Effective Training Techniques](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html)
```python
callbacks = [
    LearningRateMonitor(logging_interval="step"),
    StochasticWeightAveraging(swa_lrs=1e-2),
    GradientAccumulationScheduler(scheduling={...}),
    early_stopping
]

trainer = Trainer(
    gradient_clip_val=0.5,
    devices='auto',  # default
    logger=wandb_logger,...
)
```
--------------------
- Manage Experiment: [WandB](https://wandb.ai/site)
Weights and Biases
<!-- todo add img -->

--------------------
- Accumulate Gradients: 
Accumulated gradients run K small batches of size N before doing a backward pass, resulting a KxN effective batch size.
![Accumulate Gradients](../img/AccumulateGradients.webp)
Control batch size, improve the stability and generalisation of the model
<!-- Increasing the batch size without increasing the memory overhead. Also, the gradient accumulation technique can help us reduce the variance of gradient descent and improve the stability and generalisation of the model. -->

--------------------
- Gradient Clipping: 
Gradient clipping can be enabled to avoid exploding gradients. 

--------------------
- Stochastic Weight Averaging: 
Stochastic Weight Averaging (SWA) can make your models generalize better at virtually no additional cost. This can be used with both non-trained and trained models. The SWA procedure smooths the loss landscape thus making it harder to end up in a local minimum during optimization.
<!-- �����԰���ģ�͸��õط�����ͬʱ����Ҫ�����ѵ���ɱ���SWA �����ڷǳ����ģ�ͺ����ݼ���������Ч�ı���ģ������ֲ���Сֵ�����ģ�͵ķ���������

������˵��SWA ����ͨ����������ͬʱ����ģ��Ȩ�ص�ƽ��ֵ�������һ��ƽ��Ȩ�أ��Ӷ����һ������ƽ������ʧ�����������ģ�͵ķ�����������ѵ�������У�SWA �����������Եؼ���ģ��Ȩ�ص�ƽ��ֵ���������ƽ��Ȩ�����ں�����Ԥ�⡣

SWA �������ŵ����ڣ�������Ҫ���Ӷ����ѵ���ɱ�����ΪȨ��ƽ��������ѵ����������м��㣬������Ҫ��ģ�ͽ�������ѵ�������⣬SWA ����������Ч��ƽ����ʧ�������Ӷ�����ģ������ֲ���Сֵ�ķ��գ����ģ�͵ķ��������� -->

<!-- ��Ҫע����ǣ�SWA ������Ҫ��ѵ����������м��㣬��˿�����Ҫһ���Ķ������ʱ�䡣ͬʱ��SWA ��������һЩ�ض���ģ�ͺ����ݼ�����Ч���������ԡ���ˣ���ʹ�� SWA ����ʱ����Ҫ���ݾ����������е������Ż��� -->

--------------------


## Results
<!-- todo img -->



--------------------


--------------------

# Thanks for your listening!
