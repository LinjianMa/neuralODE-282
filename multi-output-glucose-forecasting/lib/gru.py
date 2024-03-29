
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F

import math

class GRUODEfunc(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUODEfunc, self).__init__()

        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.nfe = 0

    def forward(self, t, hidden):
        self.nfe += 1

        #gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        
        #gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        
        #i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        
        resetgate = F.sigmoid(h_r)
        inputgate = F.sigmoid(h_i)
        newgate = F.tanh(resetgate * h_n)
        
        hy = (1-inputgate) * (newgate-hidden) 
        return hy

class GRUODECell(nn.Module):

    """
    An implementation of GRUCell.
    """

    def __init__(self, input_size, hidden_size, bias=True, args=None):
        super(GRUODECell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.odefunc = GRUODEfunc(input_size, hidden_size, bias)
        self.reset_parameters()
        self.args = args
        self.integration_time = torch.tensor([0, 1]).float()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x):
        # print(x.shape)
        x = x.view(-1, x.size(2))

        if self.args.adjoint:
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        self.integration_time = self.integration_time.type_as(x)
        out = odeint(
            self.odefunc,
            x,
            self.integration_time,
            rtol=self.args.tol,
            atol=self.args.tol,
            method=self.args.method,
            )
        return out[1][None,:]

    @property
    def nfe(self):
        return self.odefunc.nfe
    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class GRUCell(nn.Module):

    """
    An implementation of GRUCell.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        # print(x.shape)
        x = x.view(-1, x.size(2))
        
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (hidden - newgate)
        return hy, hy

# class GRUModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
#         super(GRUModel, self).__init__()
#         # Hidden dimensions
#         self.hidden_dim = hidden_dim
#         # Number of hidden layers
#         self.layer_dim = layer_dim  
#         self.gru_cell = GRUCell(input_dim, hidden_dim, layer_dim)
#         self.fc = nn.Linear(hidden_dim, output_dim)
    
#     def forward(self, x):
        
#         # Initialize hidden state with zeros
#         #######################
#         #  USE GPU FOR MODEL  #
#         #######################
#         #print(x.shape,"x.shape")100, 28, 28
#         if torch.cuda.is_available():
#             h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
#         else:
#             h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

#         outs = []
#         hn = h0[0,:,:]
        
#         for seq in range(x.size(1)):
#             hn = self.gru_cell(x[:,seq,:], hn) 
#             outs.append(hn)
#         out = outs[-1].squeeze()
#         out = self.fc(out) 
#         # out.size() --> 100, 10
#         return out

