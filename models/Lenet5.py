import torch
import torch.nn as nn
import torch.nn.functional as F
from binarized_modules_stage1 import *

class LeNet5(nn.Module):
    def __init__(self,num_class):
        super(LeNet5, self).__init__()
        #self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,bias=False)
        self.conv1 = BinarizeConv2d(in_channels=1, out_channels=6, kernel_size=5,bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = BinarizeConv2d(in_channels=6, out_channels=16, kernel_size=5,bias=False)
        self.fc1 = bilinear(in_features=16*5*5, out_features=120,bias=False)
        self.fc2 = bilinear(in_features=120, out_features=84,bias=False)
        self.fc3 = bilinear(in_features=84, out_features=num_class,bias=False) # one linear layer

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x