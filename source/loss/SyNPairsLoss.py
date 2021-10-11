import torch
from torch import nn

class SyNPairsLoss(nn.Module):

    def __init__(self, name):
        super(SyNPairsLoss, self).__init__()
        self.name = name

    def forward(self, r1, r2):
        """
        Computes the N-Pairs Loss between the r1 and r2 representations.
        :param r1: Tensor of shape (batch_size, representation_size)
        :param r2: Tensor of shape (batch_size, representation_size)
        :return: he scalar loss
        """

        scores = torch.matmul(r1, r2.t())
        diagonal_mean = torch.mean(torch.diag(scores))
        r_lse = torch.mean(torch.logsumexp(scores, dim=1))
        c_lse = torch.mean(torch.logsumexp(scores, dim=0))
        return 1/2 * (r_lse - diagonal_mean) +\
               1/2 * (c_lse - diagonal_mean)


