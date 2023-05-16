# Graph Attention ViT

In pratice, since transformer becomes more and more popular, many companies and researchers want to add attention block to their model to improve performance. However, training an attention related model is computation consuming and not easy. One solution is that, froze pretrained model and only train the attention block. 

In this project, we add the attention block to the resnet18. Instead of using traditional attention block, we use graph attention block to extract the features from image patches. And we also follow the ViT structure to utilize the attention block.

-----------------

# The basis of attention

This part is about math of attention mechanism. Since it's quite popular, I think most of you have already known it. So let's skip this part.

-----------------

## Graph Attention Block

Actually, the graph attention is quite simple. It just use the linear transformation and activation function like MLP does and finally use softmax to normalize the attention.

The formula is shown here:

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

With help of matrix form, then we can implement graph attention in pytorch easily.

```python
Q = H @ W_Q
# Q : (batch, nodes, 1)
K = H @ W_K
# K : (batch, nodes, 1)
A = torch.softmax(self.activation(Q.transpose(-1,-2)+K),dim=-1)
# A : (batch, nodes, nodes)
```

Comparing to the traditional attention, the graph attention uses outer addition between Q and K instead of matrix multiplication, which reduces computation. And also it uses a layer of MLP instead of bilinear transformation, which provides more flexibility.

-----------------

## Conv2d Embedding

In order to use the graph attention, we should convert patches into feature vectors of nodes in the graph.
In this project, we use convolution to convert instead of flatten. The implementation is simple.
Just like this.

```python
nn.Conv2d(chans,feats,kernel_size=patch_size,stride=patch_size)
```

Just set the `kernel_size` and `stride_size` equal to `patch_size` to get feature vector of each patch.

-----------------

## Experiment

In experiment. We use the resnet18 as backbone and add graph attention. We use the cifar10 dataset to train the model.

| Model                     | Pretrained | Attention | Epoch Consuming | Test Accuracy |
|:-------------------------:|:----------:|:---------:|:---------------:|:-------------:|
| Resnet18                  |            |           | 50              | 0.926         |
| Resnet18 + GraphAtten     |            | ✔️        | 50              | 0.918         |
| Resnet18 \|> GraphAtten   | ✔️         | ✔️        | 5               | **0.935**     |
| Resnet18 \|> ClassicAtten | ✔️         | ✔️        | 5               | 0.931         |

From the table, we can see test accuracy of resnet18 is about 92.6%. When we add attention and train the whole model from the beginning, unfortunately, test accuracy decreases. However, when we use pretrained resnet18 and only train the attention block, test accuracy increases to 93.5%.

And we also compare the graph attention block with the traditional attention block. Test accuracy of graph attention is little higher than classical attention. 

-----------------

## Conclusion

In conclusion, using pretrained model to boost performance of new added attention is a good choice.

Here are the advantages of this method:

1. Do not need to train the whole model from the beginning. It's easy to just train a new added block.
2. Improve the performance of the existed model just in a few epochs, saving time and money.

--------------------

# Thanks for your listening!
