import numpy as np
import argparse
import logging
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math

class Spiral(nn.Module):

    def __init__(self, A):
        super(Spiral, self).__init__()
        self.A = A
    def forward(self, t, y):
        return torch.mm(y**3, self.A)

class Spiral_fit(nn.Module):

    def __init__(self):
        super(Spiral_fit, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
    def forward(self, t, y):
        return self.net(y**3)


class LinearODEF(nn.Module):
    def __init__(self, W):
        super(LinearODEF, self).__init__()
        self.lin = nn.Linear(2, 2, bias=False)
        self.lin.weight = nn.Parameter(W)
    def forward(self, t, x):
        return self.lin(x)

class LinearFunc(LinearODEF):
    def __init__(self):
        super(LinearFunc, self).__init__(
            Tensor([[-0.1, -1.], [1., -0.1]]))

class LinearFunc_fit(LinearODEF):
    def __init__(self):
        super(LinearFunc_fit, self).__init__(
            torch.randn(2, 2)/2.)


class Spiral_NN(nn.Module):
    def __init__(self, A, B, x0):
        super(Spiral_NN, self).__init__()
        self.A = nn.Linear(2, 2, bias=False)
        self.A.weight = nn.Parameter(A)
        self.B = nn.Linear(2, 2, bias=False)
        self.B.weight = nn.Parameter(B)
        self.x0 = nn.Parameter(x0)

    def forward(self, t, x):
        xTx0 = torch.sum(x*self.x0, dim=1)
        dxdt = torch.sigmoid(xTx0) * self.A(x - self.x0) + torch.sigmoid(-xTx0) * self.B(x + self.x0)
        return dxdt


class Spiral_NN_fit(nn.Module):
    def __init__(self, in_dim, hid_dim, time_invariant=False):
        super(Spiral_NN_fit, self).__init__()
        self.time_invariant = time_invariant

        if time_invariant:
            self.lin1 = nn.Linear(in_dim, hid_dim)
        else:
            self.lin1 = nn.Linear(in_dim+1, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, in_dim)
        self.elu = nn.ELU(inplace=True)

    def forward(self, t, x):
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)

        h = self.elu(self.lin1(x))
        h = self.elu(self.lin2(h))
        out = self.lin3(h)
        return out


