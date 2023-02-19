import torch
import torch.nn as nn
import torch.nn.functional as F

class TReLU(nn.Module):
    def __init__(self):
        super(TReLU, self).__init__()

    def forward(self, x):
        split_x = torch.split(x, x.shape[1]//2, dim=1)
        return torch.cat((F.relu(split_x[0]),split_x[1]),dim=1)

class LeNet5(nn.Module):
    def __init__(self,num_class):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5,bias=False)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_class) # one linear layer

    def forward(self, x):
        x = self.pool(TReLU()(self.conv1(x)))
        x = self.pool(TReLU()(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = TReLU()(self.fc1(x))
        x = TReLU()(self.fc2(x))
        x = TReLU()(self.fc3(x))
        return x