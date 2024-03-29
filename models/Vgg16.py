import torch.nn as nn
from binarized_modules_stage1 import *

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            BinarizeConv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            BinarizeConv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            BinarizeConv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            BinarizeConv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            BinarizeConv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            BinarizeConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            BinarizeConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            BinarizeConv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Tanh(),
            BinarizeConv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Tanh(),
            BinarizeConv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            BinarizeConv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Tanh(),
            BinarizeConv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Tanh(),
            BinarizeConv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            bilinear(512, 4096),
            nn.Tanh(),
            nn.Dropout(0.5),
            bilinear(4096, 4096),
            nn.Tanh(),
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.features(x)
        # print(out.shape)
        out = torch.flatten(out,1)
        # print(out.shape)
        out = self.classifier(out)
        return out