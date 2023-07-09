# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 00:41:26 2023

@author: liujy
"""

import sys
import torch
import pickle
sys.path.insert(1, './model_final')

with open('biology_aware.pkl', 'rb') as f:
    res = pickle.load(f)

data = torch.load('data.pt')
trainSet, valSet, testSet = data
# import data from file
# Data.x is gene mutation with 296 binary values, using NSCLC patients as an example
# Data.edges is obtained from ppi graph
# Data.y has first column as survival status and second column as overall survival
#%%
from model import Net
from torch_geometric.loader import DataLoader
from loss import _estimate_concordance_index, _estimate_concordance_index_true
from util import survivalStatus, survivalLength

#load data into dataloader
dl_train = DataLoader(dataset = trainSet, batch_size = 64, shuffle = True, drop_last=True)
dl_val = DataLoader(dataset = valSet, batch_size = 128, shuffle = False, drop_last=False)
dl_test = DataLoader(dataset = testSet, batch_size = 128, shuffle = False, drop_last=False)
            
# input biology knowledge to initiate the model
model = Net(res['embMat'],res['centralityMat'] ,res['paddingMask'],res["shortestPath"],res['geneLength']).double()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
val_loss = 50
for epoch in range(500):  # loop over the dataset multiple times
    model.train()

    for i, data in enumerate(dl_train, 0):
        outputs,_ = model(data.x)
        # _estimate_concordance_index is a smooth version of _estimate_concordance_index_true
        loss = 1 - _estimate_concordance_index(survivalStatus(data.y)
                                               ,survivalLength(data.y)
                                               ,torch.squeeze(outputs))
        l2_reg = torch.tensor(0.)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += 0.01 * l2_reg
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        with torch.no_grad():
            for i, data in enumerate(dl_val, 0):
                outputs,_ = model(data.x)
                loss = 1 - _estimate_concordance_index(survivalStatus(data.y)
                                                      ,survivalLength(data.y)
                                                      ,torch.squeeze(outputs))  
                if loss < val_loss and epoch > 5:
                        val_loss = loss
                        torch.save(model.state_dict(), "temp")
                        # save the weight with highest validation accuracy
                        print('success')
                        print(loss)
        
        

model.load_state_dict(torch.load("temp"))
model.eval()
with torch.no_grad():
    for i, data in enumerate(dl_test,0):
            outputs, attnWeights = model(data.x)
            loss = _estimate_concordance_index_true(survivalStatus(data.y)
                                                   ,survivalLength(data.y)
                                                     ,torch.squeeze(outputs))
            print('test loss is ++++++++++++++++++++++++++++++++++++++++++++++++++++++++' )
            print( loss)

            
