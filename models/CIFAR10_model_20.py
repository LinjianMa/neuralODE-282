from __future__ import print_function
import numpy as np
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import datasets, transforms

logger = logging.getLogger('neuralODE')


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConcatConv2d(nn.Module):

    def __init__(
            self,
            dim_in,
            dim_out,
            ksize=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True,
            transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ODEfunc(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(ODEfunc, self).__init__()
        self.conv1 = ConcatConv2d(inplanes, planes, 3, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConcatConv2d(planes, planes, 3, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.conv1(t, x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class BasicODEBlock(nn.Module):
    expansion = 1

    def __init__(self, odefunc, args, inplanes, planes, stride=1, downsample=None):
        super(BasicODEBlock, self).__init__()
        self.downsample = downsample
        self.args = args

        if downsample is not None:
            self.basicblock = BasicBlock(inplanes, planes, stride, downsample)
        else:
            self.odefunc = odefunc
            self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        if self.args.adjoint:
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        if self.downsample is not None:
            return self.basicblock.forward(x)
        else:   
            self.integration_time = self.integration_time.type_as(x)
            out = odeint(
                self.odefunc,
                x,
                self.integration_time,
                rtol=self.args.tol,
                atol=self.args.tol
                )
            return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe
    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

ALPHA_ = 1

class ResODENet(nn.Module):

    def __init__(self, depth, is_odenet, args, num_classes=10):
        super(ResODENet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        self.args = args
        self.is_odenet = is_odenet
        
        if not is_odenet:
            block = Bottleneck if depth >= 44 else BasicBlock
        else: 
            # TODO: bottleneck for ODE needs to be implemented
            block = Bottleneck if depth >= 44 else BasicODEBlock

        self.inplanes = 16 * ALPHA_
        self.conv1 = nn.Conv2d(3, 16 * ALPHA_, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16 * ALPHA_)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16 * ALPHA_, n)
        self.layer2 = self._make_layer(block, 32 * ALPHA_, n, stride=2)
        self.layer3 = self._make_layer(block, 64 * ALPHA_, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * ALPHA_ * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.Sequential_layer = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            *self.layer1,
            *self.layer2,
            *self.layer3,
            self.avgpool,
            Flatten(),
            self.fc,
            )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if not self.is_odenet:
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
        else:
            # print(self.inplanes, planes)
            layers.append(
                block(
                    ODEfunc(self.inplanes,planes,stride),
                    self.args,
                    self.inplanes, 
                    planes, 
                    stride, 
                    downsample
                    )
                )
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(
                    block(
                        ODEfunc(self.inplanes,planes),
                        self.args,
                        self.inplanes, 
                        planes
                        )
                    )            

        return layers


def CIFAR10_model_20(args):

    is_odenet = args.network == 'odenet'
    
    # TODO: here I just hard code the layer number that performs ODE
    return ResODENet(
        depth=20, 
        is_odenet=is_odenet, 
        args=args,
        num_classes=10
        ).Sequential_layer, [3,4,5,7,8,10,11]


