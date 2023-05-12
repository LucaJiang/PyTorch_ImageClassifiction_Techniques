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
        - output shape : (batch, nodes, heads * outfeats)
    """
        super().__init__()
        self.v_transform = nn.Linear(infeats,outfeats)
        self.q_transform = nn.Linear(outfeats,heads)
        self.k_transform = nn.Linear(outfeats,heads)
        self.doprout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=3)
        self.activation = activation

    def forward(self, X):
        # X : (batch, nodes, infeats)
        V = self.v_transform(X)
        # V : (batch, nodes, outfeats)
        Q = self.q_transform(V)
        # Q : (batch, nodes, heads)
        K = self.k_transform(V)
        # K : (batch, nodes, heads)
        A = self.activation(torch.einsum("bih,bjh->bijh",Q,K))
        # A : (batch, nodes, nodes, heads)
        A = self.doprout(self.softmax(A))
        H = torch.einsum("bijh,bjf->bihf",A,V)
        # H : (batch, nodes, heads, outfeats)
        O = torch.flatten(H,2)
        # O : (batch, nodes, heads * outfeats)
        return O
    
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
    def __init__(self,infeats,hidfeats,outfeats,heads=8,dropout=0.1,activation=nn.LeakyReLU()):
        """
        Graph Attention ViT Encoder layer
        -------------------------------
        ## Arguments:
        - infeats : input features
        - hidfeats : hidden features
        - outfeats : output features
        - heads : number of heads
        - dropout : dropout rate
        - activation : activation function
        -------------------------------
        ## Shape:
        - input shape : (batch, nodes, infeats)
        - output shape : (batch, nodes, outfeats + hidfeats * heads + infeats)
        """
        super().__init__()
        self.norm = nn.LayerNorm()
        self.mhga = MultiHeadGraphAtten(infeats,hidfeats,heads,dropout,activation)
        self.mlp = nn.Linear(hidfeats * heads + infeats,outfeats)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self,X):
        # X : (batch, nodes, infeats)
        H = self.norm(X)
        H = self.mhga(H)
        # H : (batch, nodes, hidfeats * heads)
        H = torch.concat([H,X],dim=2)
        # H : (batch, nodes, hidfeats * heads + infeats)
        O = self.norm(H)
        O = self.mlp(O)
        O = self.dropout(self.activation(O))
        # O : (batch, nodes, outfeats)
        O = torch.concat([O,H],dim=2)
        # O : (batch, nodes, outfeats + hidfeats * heads + infeats)
        return O

if __name__ == "__main__":
    X = torch.randn(10,3,32,32)
    embedding = Conv2dEmbed(3,6,patch_size=8,width=32,height=32)
    gal = MultiHeadGraphAtten(6,16,heads=8)
    X = embedding(X)
    assert X.shape == (10,16,6)
    X = gal(X)
    assert X.shape == (10,16,128)
    print("shape test pass")