from __future__ import print_function
from torch.autograd import Variable
import torch.nn as nn
import torch

# ------------------------------inputsize == 32 * 32
class encoder(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(encoder, self).__init__()

        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.fc1 = nn.Linear(16 * 16 * 32, 256)
        self.fc2 = nn.Linear(256, 16)
        self.relu = nn.ReLU()

    def forward(self, x):
        noise = torch.rand(x.shape) * x.mean() / 10
        x = x + noise.cuda()
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        h1 = self.relu(self.fc1(out))
        return self.fc2(h1)

class decoder(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(decoder, self).__init__()
        self.fc3 = nn.Linear(16, 256)
        self.fc4 = nn.Linear(256, 8192)
        self.relu = nn.ReLU()

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

    def forward(self, x):
        h3 = self.relu(self.fc3(x))
        out = self.relu(self.fc4(h3))
        out = out.view(out.size(0), 32, 16, 16)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        return out

class classifier(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(classifier, self).__init__()
        self.fc5 = nn.Sequential(nn.ReLU(), nn.Linear(16, out_channel))

    def forward(self, x):
        label = self.fc5(x)
        return label
