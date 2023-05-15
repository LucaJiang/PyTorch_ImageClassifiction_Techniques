# Title
author1,author2,author3,author4

--------------------

## Attention Block

In pratice, since the transformer becomes popular, many companies and researchers want to add attention block to their existed model in order to improve the performance. However, training a attention related model is computation consuming. One solution is that fix the weight of the pretrained model and only train the attention block. 

In this project, we add the attention block to the resnet18 model. Instead of using the traditional attention block, we use the graph attention block, inspired by the graph attention network, to extract the features from image spatches. And we follow the ViT structure to utilize the attention block.
<center><img src="https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/639b1df59b5ec8f6e5fdb8cf_transformer%20gif.gif" style="zoom:30%;"></center>

-----------------

### Graph Attention Block

The graph attention block is shown in the following figure,
<center><img src="https://pic4.zhimg.com/v2-526634b065899482bbe9811af105ab73_b.jpg"></center>
<center>Fig. Graph Attention Block</center>

Let $h_i$ be the input features for node $i$ and $W$ a learnable weight matrix. The self-attention mechanism computes the attention coefficients $\alpha_{ij}$ for each pair of nodes $i$ and $j$:

$$
\alpha_{ij}=softmax(\sigma(W^{\top}[h_i||h_j]))
$$

where $\sigma$ represents for activation function.

<!-- Comparing to the traditional attention block, the graph attention block is more flexible and uses less trainable parameter. -->

-----------------

### Conv2d Embedding

In order to use the graph attention block, we should convert patches into feature vectors of nodes in the graph.
In this project, we use the `Conv2d` layer to convert instead of the `Flatten` layer. 

```python
nn.Conv2d(chans,feats,kernel_size=patch_size,stride=patch_size)
```

Set the `kernel_size` and `stride_size` equal to `patch_size`.

-----------------

### Experiment

<img src="img/resvit-result.png">

| Model | Epoch Consuming | Test Accuracy |
| :---: | :---: | :---: |
| Resnet18 | 50 | 0.926 |
| Resnet18 + GraphAtten | 50 | 0.918 |
| Resnet18(pretrained) + GraphAtten| 5 | 0.935 |
| Resnet18(pretrained) + ClassicAtten| 5 | 0.931 |


The test accuracy even decreases when we add the attention block to the resnet18 model and train whole model from the begining. However, when we use the pretrained model, the metric increases.

-----------------

### Conclusion

In conclusion, using the pretrained model to boost the performance of new added attention block is a good choice.

Here are the advantages of this method:

1. Don't need to train the whole model from the beginning. It's easy to just train a new added block.
2. Improve the performance of the existed model just in a few epochs, saving time and money.

--------------------

# Thanks for your listening!
