## Why lightning

First, we can see it's easy to convert the pytorch code in the left panel to the lightning code in the right panel.

Second, it's easier to customize the training process with Trainer and Callbacks modules. You can just add the techniques you want.

Besides, Lightning includes features such as multi-GPU training and distributed training if we want to use a larger dataset or more complex model.

Therefore, we try to use lightning.

## Datasets and models (opt)
We use the CIFAR-10 dataset and ResNet18 or 34 model. NLL loss, which is the same as CrossEntropyLoss in this case. The optimizer is SGD, also we tried a commonly used Adam. 

## Transforms: Data Augmentation
CIFAR-10 is a small dataset, with 50k training images and 10k test images. It's easy to overfit and really difficult to obtain a test acc over 90%. We also applied data augmentation to reduce overfitting.

We use random crop and random rotation to augment the dataset. It's a easy but powerful tool. A study said it can at least increase 3% acc. So, as long as it makes sense, I will use it.

## Transforms: Data Normalization and Resizing
Normalize the dataset as usual.
The common practice is to resize the img to fit the input layer of the model. We try to change the input layer of our model. It's all the same.

## Transfer Learning
We use the pretrained model in ImageNet dataset. It's also a easy but powerful tool, especially when the dataset is small.

## Replicability and Determinism
We set the random seed to make our result reproducible. Also, we need to set the hard ware, but it may slow down the training process.

## lr-finder
lr-finder can help to find a suitable learning rate. If it works, we don't need to tune lr. That is if your lr finder plot looks like the blue curve, you can choose the lr in the red circle. It should be convex, but not to pick the lowest loss, but in the middle of the sharpest downward slope, which should let loss decrease faster.

## Effective Training Techniques
We use the following tricks to train and tune the model. We don't have to know the details of these tricks, but use them easily with lightning. It's worth mentioning that we can stop the training process at any time with early stopping and customize the learning rate with learning rate scheduler.

(opt)
* Gradient Accumulation: Accumulate the gradient for several batches. This can make the model converge faster. 
* Early Stopping: Stop training when the validation loss stops decreasing. This can prevent overfitting. 
* Gradient Clipping: Clip the gradient to a certain range. Prevent gradient explosion or vanish.
* Stochastic Weight Averaging: Average the weights of the model during training.
* Learning Rate Scheduler: Customize and control learning rate. This can make the model converge faster. 

## Results
One of the greatest tool is wandb. It can automatically log the training process and show the results in a beautiful way. It's really convenient to use. 

We can find the dash board in this link. It easy to compare the results of different models.

The pink curve is used the lr recommend by lr finder. It's too large so the loss dropped quickly at the beginning and then get stuck in a local minimum. 

The green curve divided the lr by 10. The loss slowly decreased to nearly 0. But this model still a little overfitting. The val and test loss are a bit higher than training.
