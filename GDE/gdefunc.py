import dgl
import torch
import torch.nn as nn
from .gcn import GCNLayer

from typing import Callable


class GCDEFunc(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, graph:dgl.DGLGraph, activation:Callable[[torch.Tensor], torch.Tensor],
                 dropout:int):
        """Standard GCDN ODE function class. To be passed to an ODEBlock"""
        super().__init__()
        self.l = GCNLayer(graph, input_dim, hidden_dim, activation=activation, dropout=dropout)
        self.nfe = 0
        self.g = graph
    
    def set_graph(self, g:dgl.DGLGraph):
            self.l.g = g
            
    def forward(self, t, x):
        self.nfe += 1
        x = self.l(x)
        return x
    
class ContGCDEFunc(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, graph:dgl.DGLGraph, activation:Callable[[torch.Tensor], torch.Tensor], 
                 dropout:int):
        """ Controlled GCDN version. Input information is preserved longer via hooks to input node features X_0
          present at all ODE function steps"""
        super().__init__()
        self.l = GCNLayer(graph, input_dim, hidden_dim, activation=activation, dropout=dropout)
        self.controller = GCNLayer(graph, input_dim, hidden_dim, activation=activation, dropout=dropout)
        self.nfe = 0
        self.g = graph
    
    def set_graph(self, g):
            self.l.g = g
            
    def forward(self, t, x):
        self.nfe += 1
        x = self.l(x[:,:,0]) + self.controller(x[:,:,1])
        return x
    