import torch.nn as nn


# -----------------------input size>=32---------------------------------
class LeNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel, 6, 5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(6, 16, 5),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d((5))  # adaptive change the outputsize to (16,5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5, 30),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(30, 10),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(10, out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x