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
        Q = self.q_transform(V).transpose(1,2)
        # Q : (batch, heads, nodes)
        K = self.k_transform(V).transpose(1,2)
        # K : (batch, heads, nodes)
        A = self.activation(torch.einsum("bhi,bhj->bhij",Q,K))
        # A : (batch, heads, nodes, nodes)
        A = self.doprout(self.softmax(A))
        H = torch.einsum("bhij,bjf->bihf",A,V)
        # H : (batch, nodes, heads, outfeats)
        O = torch.flatten(H,2)
        # O : (batch, nodes, heads * outfeats)
        return O
    
class Conv2dEmbed(nn.Module):
    def __init__(self,chans,feats,patch_size=1):
        """
        Conv2d Embedding layer
        -------------------------------
        ## Arguments:
        - chans : number of input channels
        - feats : number of output features
        - patch_size : patch size
        -------------------------------
        ## Shape:
        - input shape : (batch, channels, height, width)
        - output shape : (batch, nodes, features)
        """
        super().__init__()
        self.conv = nn.Conv2d(chans,feats,kernel_size=patch_size,stride=patch_size)
        self.flatten = nn.Flatten(2)

    def forward(self,X):
        # X : (batch, channels, height, width)
        X = self.conv(X)
        # X : (batch, features, height // patch_size, width // patch_size)
        X = self.flatten(X)
        # X : (batch, features, nodes)
        X = torch.transpose(X,1,2)
        # X : (batch, nodes, features)
        return X
    
if __name__ == "__main__":
    X = torch.randn(10,3,32,32)
    embedding = Conv2dEmbed(3,6,patch_size=8)
    gal = MultiHeadGraphAtten(6,16,heads=8)
    X = embedding(X)
    assert X.shape == (10,16,6)
    X = gal(X)
    assert X.shape == (10,16,128)
    print("shape test pass")