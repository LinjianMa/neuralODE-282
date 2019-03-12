from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class QAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(QAlexNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=5, stride=1, padding=2)  # 32x32x3 -> 32x32x64
        # self.pool1=nn.MaxPool2d(kernel_size=3, stride=2, padding =1 )# 32x32x64
        # -> 16x16x64
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(
            64, 64, kernel_size=5, stride=1, padding=2)  # 16x16x64 -> 16x16x64
        # self.pool2=nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)# 16x16x64
        # -> 8x8x64
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)

    def squeeze_layers(self, sl=None):
        for k in self._modules.keys():
            if k in sl:
                for param in self._modules[k].parameters():
                    param.requires_grad = False
                    print(param.requires_grad)

    def back(self):
        for k in self._modules.keys():
            for param in self._modules[k].parameters():
                param.requires_grad = True

    def forward(self, x):
        x = F.max_pool2d(self.bn1(F.relu(self.conv1(x))), 3, 2, 1)
        x = F.max_pool2d(self.bn2(F.relu(self.conv2(x))), 3, 2, 1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


ALPHA = 1


class ANet(nn.Module):
    def __init__(self):
        super(ANet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64 * ALPHA, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(64 * ALPHA)
        self.conv2 = nn.Conv2d(64 * ALPHA, 64 * ALPHA, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64 * ALPHA)
        self.conv3 = nn.Conv2d(64 * ALPHA, 128 * ALPHA, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128 * ALPHA)
        self.conv4 = nn.Conv2d(128 * ALPHA, 128 * ALPHA, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(128 * ALPHA)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128 * 5 * 5 * ALPHA, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.drop = nn.Dropout()

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = F.max_pool2d(self.bn2(F.relu(self.conv2(x))), 2)
        x = self.bn3(F.relu(self.conv3(x)))
        x = F.max_pool2d(self.bn4(F.relu(self.conv4(x))), 2)
        #x = self.conv_drop(x)
        x = x.view(-1, 128 * ALPHA * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            # 32x32x3 -> 8x8x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8x64 -> 4x4x64
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # 4x4x64 -> 4x4x192
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4x64 -> 2x2x192
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # 2x2x192 -> 2x2x384
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # 2x2x384 -> 2x2x256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # 2x2x256 -> 2x2x256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2x256 -> 1x1x256
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.drop = nn.Dropout()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = self.conv_drop(x)
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x


class FcNet(nn.Module):
    def __init__(self, num_classes=10):
        super(FcNet, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 2048)
        self.bn1 = nn.BatchNorm2d(2048)
        self.fc2 = nn.Linear(2048, 256)
        self.bn2 = nn.BatchNorm2d(256)
        self.fc3 = nn.Linear(256, 10)
        self.drop = nn.Dropout()

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.drop(x)
        x = self.fc3(x)
        return x


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
