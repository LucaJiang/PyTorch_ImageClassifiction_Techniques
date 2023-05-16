# Title

author1,author2,author3,author4

--------------------

# Graph Attention ViT

In pratice, since the transformer becomes more and more popular, many companies and researchers want to add **attention** block to their existed model in order to improve the performance. However, training an attention related model is computation consuming and not easy.

In this project, we add the attention block to the resnet18 model. And we follow the ViT structure to utilize the attention block.

<center><img src="https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/639b1df59b5ec8f6e5fdb8cf_transformer%20gif.gif" style="zoom:30%;"></center>

-----------------

## The basis of attention

### The general form for the self-attention

The general form for self-attention is as follows:

$$
A_{ij}=f(h_i,h_j)
$$

where $h_i$ and $h_j$ are the features for node $i$ and $j$, and $f$ is an arbitrary function that computes the attention score between two nodes.

### The classical self-attention

The classical self-attention is computed as follows:

$$
A=softmax\left(\frac{H^{\top}(Q^{\top}K)H}{\sqrt{d_k}}\right)
$$

where $H\in\mathbb{R}^{f\times n}$ is the feature matrix for each embedding, $Q,K\in\mathbb{R}^{f^{'}\times f}$ are the query and key matrix for self-attention, and $d_k$ is the dimension of the key vector. Actually, it's just a bilinear function. 

-----------------

## Graph Attention Block

The graph attention block is shown in the following figure,

<center><img src="https://pic4.zhimg.com/v2-526634b065899482bbe9811af105ab73_b.jpg" style="zoom:.8"></center>

### The formula for graph attention

$$
\alpha_{ij}=softmax(\sigma(W^{\top}[h_i||h_j]))
$$

where $W\in\mathbb{R}^{f\times 1}$ is the weight, $h_i\in \mathbb{R}^{f}$ is the feature of the $i^{th}$ node  and $\sigma$ represents for activation function.

-----------------

## Graph Attention Block

Convert in this form, the attention matrix $A$ can be computed as follows:

$$
A=softmax(\sigma(W_K^{\top}H+H^{\top}W_Q))
$$

where $H\in\mathbb{R}^{f\times n}$ is the feature matrix of all nodes.

Then we can implement the graph attention block in pytorch like this:

```python
Q = H @ W_Q
# Q : (batch, nodes, 1)
K = H @ W_K
# K : (batch, nodes, 1)
A = torch.softmax(self.activation(Q.transpose(-1,-2)+K),dim=-1)
# A : (batch, nodes, nodes)
```

<!-- Comparing to the traditional attention block, the graph attention block use addition between Q and K instead of matrix multiplication, which reduce computation -->

-----------------

## Conv2d Embedding

In order to use the graph attention block, we should convert patches into feature vectors of nodes in the graph.
In this project, we use the `Conv2d` layer to convert instead of the `Flatten` layer. 

```python
nn.Conv2d(chans,feats,kernel_size=patch_size,stride=patch_size)
```

Set the `kernel_size` and `stride_size` equal to `patch_size`.

-----------------

## Experiment

<img src="img/resvit-result.png">

| Model                     | Pretrained | Attention | Epoch Consuming | Test Accuracy |
|:-------------------------:|:----------:|:---------:|:---------------:|:-------------:|
| Resnet18                  |            |           | 50              | 0.926         |
| Resnet18 + GraphAtten     |            | ✔️        | 50              | 0.918         |
| Resnet18 \|> GraphAtten   | ✔️         | ✔️        | 5               | **0.935**     |
| Resnet18 \|> ClassicAtten | ✔️         | ✔️        | 5               | 0.931         |

The test accuracy even decreases when we add the attention block to the resnet18 model and train whole model from the begining. However, when we use the pretrained model, the metric increases.

-----------------

## Conclusion

In conclusion, using the pretrained model to boost the performance of new added attention block is a good choice.

Here are the advantages of this method:

1. Don't need to train the whole model from the beginning. It's easy to just train a new added block.
2. Improve the performance of the existed model just in a few epochs, saving time and money.

--------------------

# Thanks for your listening!
