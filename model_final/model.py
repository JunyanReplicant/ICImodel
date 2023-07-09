# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.optim as optim

class Net(nn.Module):
    def __init__(self,embMat,centralityMat,paddingMask,shortestPath, geneNum):
        super().__init__()
        self.geneNum = geneNum
        self.centralityMat = torch.tensor(centralityMat)
        self.padding = torch.tensor(paddingMask)
        self.shortestPath = torch.tensor(shortestPath).double()
        self.embMat = torch.tensor(embMat.T)
        self.seq_len, self.emb_len = self.embMat.size()
        self.embedLayer = self.create_emb_layer()
        self.linq = nn.Linear(self.emb_len, self.emb_len, bias=False)
        self.link = nn.Linear(self.emb_len, self.emb_len, bias=False)
        self.linv = nn.Linear(self.emb_len, self.emb_len, bias=False)
        self.k1 = torch.nn.Parameter(torch.randn((self.geneNum,self.geneNum))) 
        self.k2 = torch.nn.Parameter(torch.randn((self.geneNum,self.geneNum)))
        self.k3 = torch.nn.Parameter(torch.randn(1))
        self.fc1 =  nn.Linear(self.geneNum, self.geneNum)
        self.fcCox =  nn.Linear(self.geneNum, 1, bias=False)
        self.reset_parameters()
        
        
    def create_emb_layer(self):
        num_genes, embedding_dim = self.embMat.shape
        emb_layer = nn.Embedding(num_genes , embedding_dim)
        emb_layer.load_state_dict({'weight': self.embMat})
        emb_layer.weight.requires_grad = False
        return emb_layer    
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linq.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.link.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.fc1.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.fcCox.weight, gain=1 / math.sqrt(2))

        
    def forward(self, datax): 
        x = torch.reshape(datax,[-1,self.geneNum])
        originalx = x
        originalx[originalx != 0 ] = 1
        x = self.embedLayer(x)
        q = self.linq(x)
        k = self.link(x)
        v = torch.reshape(datax,[-1,self.geneNum,1]).double()
        v[v!=0] = 1
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        # # apply mask
        attn_weights = attn_weights + nn.LeakyReLU()(self.k1) * self.shortestPath + nn.LeakyReLU()(self.k2) * self.centralityMat   + nn.LeakyReLU()(self.k3) * self.padding 
        attn_weights_origin = attn_weights
        attn_weights_float = nn.Softmax(dim=-1)(attn_weights)
        attn = torch.bmm(attn_weights_float, v)
        attn = nn.Dropout(p = 0.2)(attn)
        attn = torch.squeeze(attn)
        attn = originalx + attn
        attn = self.fc1(attn)
        output = self.fcCox(attn)
        return (output.flatten()),attn_weights_origin 
