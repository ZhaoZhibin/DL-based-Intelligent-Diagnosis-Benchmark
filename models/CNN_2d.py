#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings

# ----------------------------inputsize >=28*28-------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=10):
        super(CNN, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=3),  # 16, 26 ,26
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),  # 32, 24, 24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 32, 12,12     (24-2) /2 +1

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # 64,10,10
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),  # 128,8,8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((4,4)))  # 128, 4,4

        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True))
        self.fc = nn.Linear(128, out_channel)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        x = self.fc(x)

        return x
