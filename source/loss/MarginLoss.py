import torch
from torch import nn


class MarginLoss(nn.Module):

    def __init__(self, name):
        super(MarginLoss, self).__init__()
        self.mse = nn.MSELoss()


    def cos_sim(self, r1, r2):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        r1_norm = torch.nn.functional.normalize(r1, p=2, dim=1)
        r2_norm = torch.nn.functional.normalize(r2, p=2, dim=1)
        return torch.mm(r1_norm, r2_norm.transpose(0, 1))


    def forward(self, r1, r2):
        scores = self.cos_sim(r1, r2)
        target = 2 * torch.eye(r1.shape[-1]) - 1
        return self.mse(scores, target)


