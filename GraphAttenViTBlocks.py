import torch 
from torch import nn

class MultiHeadGraphAtten(nn.Module):
    def __init__(self,infeats,outfeats,heads=8,dropout=0.1,activation=nn.LeakyReLU()):
        """
        Multi-Head Graph Attention layer
        -------------------------------
        ## Arguments:

        - infeats : input features
        - outfeats : output features
        - heads : number of heads
        - dropout : dropout rate
        - activation : activation function
        -------------------------------
        ## Shape:

        - input shape : (batch, nodes, infeats)
        - output shape : (batch, nodes, outfeats)
        """
        super().__init__()
        self.v_transform = nn.Linear(infeats,outfeats//heads)
        self.q_transform = nn.Linear(outfeats//heads,heads)
        self.k_transform = nn.Linear(outfeats//heads,heads)
        self.doprout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = activation

    def forward(self, X):
        # X : (batch, nodes, infeats)
        V = self.v_transform(X)
        # V : (batch, nodes, outfeats//heads)
        Q = self.q_transform(V).transpose(1,2)[...,None]
        # Q : (batch, heads, nodes, 1)
        K = self.k_transform(V).transpose(1,2)[...,None]
        # K : (batch, heads, nodes, 1)
        A = self.activation(Q.transpose(-1,-2)+K)
        # A : (batch, heads, nodes, nodes)
        A = self.doprout(self.softmax(A))
        H = torch.einsum("bhij,bjf->bihf",A,V)
        # H : (batch, nodes, heads, outfeats//heads)
        O = torch.flatten(H,2)
        # O : (batch, nodes, outfeats)
        return O
    
class MultiHeadAtten(nn.Module):
    def __init__(self,infeats,outfeats,heads=8,dropout=0.1,activation=nn.LeakyReLU()):
        """
        Multi-Head Classical Attention layer
        -------------------------------
        ## Arguments:

        - infeats : input features
        - outfeats : output features
        - heads : number of heads
        - dropout : dropout rate
        - activation : activation function
        -------------------------------
        ## Shape:

        - input shape : (batch, nodes, infeats)
        - output shape : (batch, nodes, heads * outfeats)
        """
        super().__init__()
        self.sqrt_d = (outfeats//heads)**0.5

        self.heads_unflatten = nn.Unflatten(2,(heads,outfeats//heads))
        self.v_transform = nn.Linear(infeats,outfeats//heads)
        self.q_transform = nn.Linear(infeats,outfeats)
        self.k_transform = nn.Linear(infeats,outfeats)

        self.doprout = nn.Dropout(dropout)
        self.activation = activation
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,X):
        # X : (batch, nodes, infeats)
        V = self.activation(self.v_transform(X))
        # V : (batch, nodes, outfeats//heads)
        Q = self.heads_unflatten(self.q_transform(X))
        K = self.heads_unflatten(self.k_transform(X))
        # Q,K : (batch, nodes, heads, outfeats//heads)
        A = torch.einsum("bihf,bjhf->bijh",Q,K)/self.sqrt_d
        # A : (batch, nodes, nodes, heads)
        A = self.softmax(A)
        H = torch.einsum("bijh,bjf->bihf",A,V)
        # H : (batch, nodes, heads, outfeats//heads)
        O = torch.flatten(H,2)
        # O : (batch, nodes, outfeats)
        return self.doprout(O)

class Conv2dEmbed(nn.Module):
    def __init__(self,chans,feats,patch_size=1,height=None,width=None):
        """
        Conv2d Embedding layer
        -------------------------------
        ## Arguments:
        - chans : number of input channels
        - feats : number of output features
        - patch_size : patch size
        - height : height of input image
        - width : width of input image
        -------------------------------
        ## Shape:
        - input shape : (batch, channels, height, width)
        - output shape : (batch, nodes, features)
        """
        super().__init__()
        self.conv = nn.Conv2d(chans,feats,kernel_size=patch_size,stride=patch_size)
        nodes = (height // patch_size) * (width // patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, nodes, feats))
        # the position embeddings are learnable parameters
        # self.position_embeddings : (1, nodes, features)

    def forward(self,X):
        # X : (batch, channels, height, width)
        X = self.conv(X)
        # X : (batch, features, height // patch_size, width // patch_size)
        X = torch.flatten(X,2)
        # X : (batch, features, nodes)
        X = torch.transpose(X,1,2)
        # X : (batch, nodes, features)
        X += self.position_embeddings
        return X
    
class GViTEncoder(nn.Module):
    def __init__(self,feats,hidden=None,heads=8,dropout=0.1,activation=nn.GELU(),attention=MultiHeadGraphAtten):
        """
        Graph Attention ViT Encoder layer
        -------------------------------
        ## Arguments:
        - feats : input features
        - heads : number of heads
        - dropout : dropout rate
        - activation : activation function
        -------------------------------
        ## Shape:
        - input shape : (batch, nodes, infeats)
        - output shape : (batch, nodes, outfeats + hidfeats * heads + infeats)
        """
        super().__init__()
        hidden = hidden or feats
        self.norm1 = nn.LayerNorm(feats)
        self.norm2 = nn.LayerNorm(feats)
        self.msa = attention(feats,feats,heads,dropout,activation)
        self.mlp = nn.Sequential(
            nn.Linear(feats, hidden),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden, feats),
            activation,
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self,X):
        # X : (batch, nodes, feats)
        X = self.msa(self.norm1(X)) + X
        return self.mlp(self.norm2(X)) + X

if __name__ == "__main__":
    X = torch.randn(10,3,32,32)
    embedding = Conv2dEmbed(3,16,patch_size=8,width=32,height=32)
    gal = MultiHeadGraphAtten(16,16,heads=8)
    al = MultiHeadAtten(16,16,heads=8)
    encoder = GViTEncoder(16,heads=4)
    H = embedding(X)
    assert H.shape == (10,16,16)
    Y = gal(H)
    assert Y.shape == (10,16,16)
    Y = al(H)
    assert Y.shape == (10,16,16)
    Y = encoder(H)
    assert Y.shape == (10,16,16)
    print("shape test pass")