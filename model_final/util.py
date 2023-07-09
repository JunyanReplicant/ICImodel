# -*- coding: utf-8 -*-

import torch

def survivalStatus(data):
    return torch.reshape(data,[-1,5])[:,0]

def survivalLength(data):
    return torch.reshape(data,[-1,5])[:,1]