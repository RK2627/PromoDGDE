# networks.py

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# for SC
class Generator_net(nn.Module):
    def __init__(self, n_chars, seq_len, batch_size, hidden):
        super(Generator_net, self).__init__()
        self.fc1 = nn.Linear(128, hidden*seq_len)
        #
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(True),
            nn.Conv1d(hidden, n_chars, 1)
        )
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden = hidden

    def forward(self, noise):
        output = self.fc1(noise)
        output = output.view(-1, self.hidden, self.seq_len)
        output = self.conv_blocks(output)
        output = output.transpose(1, 2)
        output = F.tanh(output)
        return output

class Discriminator_net(nn.Module):
    def __init__(self, n_chars, seq_len, batch_size, hidden):
        super(Discriminator_net, self).__init__()
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden = hidden
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(n_chars, hidden, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden, 1, 1)
        )
        self.linear = nn.Linear(seq_len, 1)  

    def forward(self, input):
        output = input.transpose(1, 2)
        output = self.conv_blocks(output)
        output = output.view(-1, self.seq_len)
        output = self.linear(output)
        return output

# for e.coli
# class Generator_net(nn.Module):
#     def __init__(self, n_chars, seq_len, batch_size, hidden):
#         super(Generator_net, self).__init__()
#         self.fc1 = nn.Linear(128, hidden*seq_len)
#         #
#         self.conv_blocks = nn.Sequential(
#             nn.Conv1d(hidden, hidden, 3, padding=1),
#             nn.BatchNorm1d(hidden),
#             nn.ReLU(True),
#             nn.Conv1d(hidden, hidden, 3, padding=1),
#             nn.BatchNorm1d(hidden),
#             nn.ReLU(True),
#             nn.Conv1d(hidden, n_chars, 1)
#         )
#         self.n_chars = n_chars
#         self.seq_len = seq_len
#         self.batch_size = batch_size
#         self.hidden = hidden
#
#     def forward(self, noise):
#         output = self.fc1(noise)
#         output = output.view(-1, self.hidden, self.seq_len)
#         output = self.conv_blocks(output)
#         output = output.transpose(1, 2)
#         output = F.tanh(output)
#         return output
#
# class Discriminator_net(nn.Module):
#     def __init__(self, n_chars, seq_len, batch_size, hidden):
#         super(Discriminator_net, self).__init__()
#         self.n_chars = n_chars
#         self.seq_len = seq_len
#         self.batch_size = batch_size
#         self.hidden = hidden
#         self.conv_blocks = nn.Sequential(
#             nn.Conv1d(n_chars, hidden, 3, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv1d(hidden, hidden, 3, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv1d(hidden, 1, 1)
#         )
#         self.linear = nn.Linear(seq_len, 1)
#
#     def forward(self, input):
#         output = input.transpose(1, 2)
#         output = self.conv_blocks(output)
#         output = output.view(-1, self.seq_len)
#         output = self.linear(output)
#         return output