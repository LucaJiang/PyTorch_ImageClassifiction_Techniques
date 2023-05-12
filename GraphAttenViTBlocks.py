import torch 
from torch import nn

class GraphAttenLayer(nn.Module):
    def __init__(self,infeats,outfeats,dropout=0.1,activation=nn.LeakyReLU()):
        super().__init__()
        self.v_transform = nn.Linear(infeats,outfeats)
        self.q_transform = nn.Linear(outfeats,1)
        self.k_transform = nn.Linear(outfeats,1)
        self.doprout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)
        self.activation = activation

    def forward(self, X):
        # X : (batch, nodes, infeats)
        V = self.v_transform(X)
        # V : (batch, nodes, outfeats)
        Q = self.q_transform(V)
        # Q : (batch, nodes, 1)
        K = self.k_transform(V)
        # K : (batch, nodes, 1)
        KT = torch.transpose(K,1,2)
        # KT : (batch, 1, nodes)
        A = self.softmax(self.activation(KT + Q))
        # A : (batch, nodes, nodes)
        A = self.doprout(A)
        H = torch.einsum("kij,kjf->kif",A,V)
        # H : (batch, nodes, outfeats)
        return H
    
class Conv2dEmbed(nn.Module):
    def __init__(self,chans,feats,patch_size=1):
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
    gal = GraphAttenLayer(6,16)
    X = embedding(X)
    assert X.shape == (10,16,6)
    X = gal(X)
    assert X.shape == (10,16,16)
    print("test pass")