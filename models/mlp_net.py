import torch
import torch.nn as nn

class MLPNet(nn.Module):
    def __init__(self, num_classes):
        super(MLPNet, self).__init__()

        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x

    def zero_weights(self):
        self.fc1.weight.data.fill_(0.0)
        self.fc1.bias.data.fill_(0.0)
        self.fc2.weight.data.fill_(0.0)
        self.fc2.bias.data.fill_(0.0)