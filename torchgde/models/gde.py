import dgl
import torch
import torch.nn as nn
from .gcn import GCNLayer

from typing import Callable


class GDEFunc(nn.Module):
    def __init__(self, gnn:nn.Module):
        """General GDE function class. To be passed to an ODEBlock"""
        super().__init__()
        self.gnn = gnn
        self.nfe = 0
    
    def set_graph(self, g:dgl.DGLGraph):
        for layer in self.gnn:
            layer.g = g
            
    def forward(self, t, x):
        self.nfe += 1
        x = self.gnn(x)
        return x

    
class ControlledGDEFunc(GDEFunc):
    def __init__(self, gnn:nn.Module):
        """ Controlled GDE version. Input information is preserved longer via hooks to input node features X_0, 
            affecting all ODE function steps. Requires assignment of '.h0' before calling .forward"""
        super().__init__(gnn)
        self.nfe = 0
            
    def forward(self, t, x):
        self.nfe += 1
        x = torch.cat([x, self.h0], 1)
        x = self.gnn(x)
        return x
    