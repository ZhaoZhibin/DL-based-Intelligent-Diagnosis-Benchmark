from __future__ import print_function
import torch.nn as nn

Output_size = 64
inputflag = 0
# ----------------------------------inputsize == 1024
class encoder(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(encoder, self).__init__()
        self.in_channel = in_channel
        self.fc1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))

        self.fc3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))

        self.fc4 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True))

        self.fc5 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))

        self.fc6 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.fc7 = nn.Sequential(
            nn.Linear(64, Output_size),)

    def forward(self, x):
        global inputflag
        if x.shape[2] == 512:
            inputflag = 0
            out = x.view(x.size(0), -1)
            out = self.fc1(out)
        else:
            inputflag = 1
            out = x.view(x.size(0), -1)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        out = self.fc7(out)
        return out


class decoder(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(decoder, self).__init__()
        self.in_channel = in_channel
        self.fc1 = nn.Sequential(
            nn.Linear(Output_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))

        self.fc3 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True))

        self.fc4 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))

        self.fc5 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))

        self.fc6 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))

        self.fc7 = nn.Sequential(
            nn.Linear(1024, 512),)


    def forward(self, z):
        out = self.fc1(z)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        if inputflag == 0:
            out = self.fc6(out)
            out = self.fc7(out)
        else:
            out = self.fc6(out)

        return out


class classifier(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(classifier, self).__init__()
        self.fc6 = nn.Sequential(nn.ReLU(), nn.Linear(Output_size, out_channel))

    def forward(self, z):
        label = self.fc6(z)
        return label
