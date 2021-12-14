import torch
from torch import nn


class EntropyLossEncap(nn.Module):
    def __init__(self, eps=1e-12):
        super(EntropyLossEncap, self).__init__()
        self.eps = eps

    def forward(self, input):
        b = input * torch.log(input + self.eps)
        b = -1.0 * b.sum(dim=1)
        b = b.mean()
        return b
