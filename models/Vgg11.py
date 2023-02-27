import torch
import torch.nn as nn
from binarized_modules_stage1 import *

class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__()
        self.features = nn.Sequential(
            BinarizeConv2d(3, 64, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BinarizeConv2d(64, 128, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BinarizeConv2d(128, 256, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            BinarizeConv2d(256, 256, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BinarizeConv2d(256, 512, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            BinarizeConv2d(512, 512, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            bilinear(512 * 7 * 7, 4096),
            #nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
