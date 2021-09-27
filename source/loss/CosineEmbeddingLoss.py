import torch
from torch import nn


class CosineEmbeddingLoss(nn.Module):

    def __init__(self, name):
        super(CosineEmbeddingLoss, self).__init__()
        self.loss = torch.nn.CosineEmbeddingLoss(
            margin=0.0,
            size_average=None,
            reduce=None,
            reduction='mean')
        self.name = name

    def forward(self, r1, r2, cls):
        """
        Computes the N-Pairs Loss between the r1 and r2 representations.
        :param r1: Tensor of shape (batch_size, representation_size)
        :param r2: Tensor of shape (batch_size, representation_size)
        :return: he scalar loss
        """
        return self.loss(r1,r2, cls)


