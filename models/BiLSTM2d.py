import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BiLSTM(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(BiLSTM, self).__init__()
        self.hidden_dim = 64
        self.kernel_num = 16
        self.num_layers = 2
        self.V = 5
        self.embed1 = nn.Sequential(
            nn.Conv2d(in_channel, self.kernel_num, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.kernel_num),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.embed2 = nn.Sequential(
            nn.Conv2d(self.kernel_num, self.kernel_num*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.kernel_num*2),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(self.V))
        self.hidden2label1 = nn.Sequential(nn.Linear(self.V*self.V * 2 * self.hidden_dim, self.hidden_dim * 4), nn.ReLU(), nn.Dropout())
        self.hidden2label2 = nn.Linear(self.hidden_dim * 4, out_channel)
        self.bilstm = nn.LSTM(self.kernel_num*2, self.hidden_dim,
                              num_layers=self.num_layers, bidirectional=True,
                              batch_first=True, bias=False)

    def forward(self, x):
        x = self.embed1(x)
        x = self.embed2(x)
        x = x.view(-1, self.kernel_num*2, self.V*self.V)
        x = torch.transpose(x, 1, 2)
        bilstm_out, _ = self.bilstm(x)
        bilstm_out = torch.tanh(bilstm_out)
        bilstm_out = bilstm_out.view(bilstm_out.size(0), -1)
        logit = self.hidden2label1(bilstm_out)
        logit = self.hidden2label2(logit)

        return logit
