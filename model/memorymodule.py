import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


def hard_shrink_relu(input, lambd=0.0, epsilon=1e-12):  # Eq.7
    output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output


class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.randn(self.mem_dim, self.fea_dim))  # N x C, Parameter使参数可训练
        self.shrink_thres = shrink_thres

    def forward(self, input):
        att_weight = F.linear(input, self.weight)  # (TxC) x (CxN) = TxN， F.linear带转置
        att_weight = F.softmax(att_weight, dim=1)  # w_i
        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)  # w^_i
            att_weight = F.normalize(att_weight, p=1, dim=1)
        output = F.linear(att_weight, self.weight.permute(1, 0))  # (TxN) x (NxC) = TxC
        return output, att_weight


class MemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.05):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

    def forward(self, input):
        s = input.data.shape
        input = input.reshape(s[0], -1)

        out, att = self.memory(input)
        out = out.reshape(s[0], s[1], s[2])

        return out, att
