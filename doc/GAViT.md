# Graph Attention ViT

In pratice, since the transformer becomes more and more popular, many companies and researchers want to add attention block to their existed model in order to improve the performance. However, training a attention related model is computation consuming and hard. One solution is that fix the weight of the pretrained model and only train the attention block. 

In this project, we add the attention block to the resnet18 model. Instead of using the traditional attention block, we use the graph attention block, inspired by the graph attention network, to extract the features from image patches. And we follow the ViT structure to utilize the attention block.

-----------------

# The basis of attention

This part is the mathematics of attention mechanism. Since it's quite popular, I think most of you have already known it. So let's just skip this part.

-----------------

## Graph Attention Block

Actually, the graph attention is quite simple. It just use the linear transformation and an activation function like MLP and finally use softmax to normalize the attention.

The formula is as follows:

$$
\alpha_{ij}=softmax(\sigma(W^{\top}[h_i||h_j]))
$$

where $W\in\mathbb{R}^{f\times 1}$ is the weight, $h_i\in \mathbb{R}^{f}$ is the feature of the $i^{th}$ node  and $\sigma$ represents for activation function.

-----------------

## Graph Attention Block

In order to implement the graph attention block, we should convert the formula into matrix form.

The attention matrix $A$ can be computed as follows:

$$
A=softmax(\sigma(W_K^{\top}H+H^{\top}W_Q))
$$

where $H\in\mathbb{R}^{f\times n}$ is the feature matrix of all nodes.

With the help of matrix form, then we can implement the graph attention block in pytorch easily.

```python
Q = H @ W_Q
# Q : (batch, nodes, 1)
K = H @ W_K
# K : (batch, nodes, 1)
A = torch.softmax(self.activation(Q.transpose(-1,-2)+K),dim=-1)
# A : (batch, nodes, nodes)
```

Comparing to the traditional attention block, the graph attention block use outer addition between Q and K instead of matrix multiplication, which reduce computation. And also it use a layer of MLP instead of bilinear transformation, which provides more flexibility.

-----------------

## Conv2d Embedding

In order to use the graph attention block, we should convert patches into feature vectors of nodes in the graph.
In this project, we use the covlution layer to convert instead of the flatten layer. The implementation is simple.
Just like this.

```python
nn.Conv2d(chans,feats,kernel_size=patch_size,stride=patch_size)
```

Set the `kernel_size` and `stride_size` equal to `patch_size`.

-----------------

## Experiment

Then we carry out the experiment. We use the resnet18 as the backbone and add the graph attention block to the resnet18. We use the cifar10 dataset to train the model.

| Model                     | Pretrained | Attention | Epoch Consuming | Test Accuracy |
|:-------------------------:|:----------:|:---------:|:---------------:|:-------------:|
| Resnet18                  |            |           | 50              | 0.926         |
| Resnet18 + GraphAtten     |            | ✔️        | 50              | 0.918         |
| Resnet18 \|> GraphAtten   | ✔️         | ✔️        | 5               | **0.935**     |
| Resnet18 \|> ClassicAtten | ✔️         | ✔️        | 5               | 0.931         |

From the table, we can see that the test accuracy of the resnet18 model is about 0.926. When we add the attention block to the resnet18 model and train the whole model from the beginning, the test accuracy decreases to 0.918. However, when we use the pretrained model and only train the attention block, the test accuracy increases to 0.935.

And we also compare the graph attention block with the traditional attention block. The test accuracy of the graph attention block is 0.935, while the test accuracy of the traditional attention block is 0.931. The graph attention block is a little better than the traditional attention block.

-----------------

## Conclusion

In conclusion, using the pretrained model to boost the performance of new added attention block is a good choice.

Here are the advantages of this method:

1. Do not need to train the whole model from the beginning. It's easy to just train a new added block.
2. Improve the performance of the existed model just in a few epochs, saving time and money.

--------------------

# Thanks for your listening!
