import torch.nn as nn
import torch

""" class TReLU(nn.Module):
    def __init__(self, t=0.5):
        super(TReLU, self).__init__()
        self.t = nn.Parameter(torch.tensor(t))

    def forward(self, x):
        return x * (x > self.t) + self.t * (x <= self.t) """


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.t = nn.Parameter(torch.tensor(.5,requires_grad=True))
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            #TReLU(),
            #nn.MaxPool2d(kernel_size = 2, stride = 2))
        )
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out_t = out + self.t
        out = self.maxpool(out_t)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out,out_t
