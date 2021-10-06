import torch
from torch import nn


class MNSRankingLoss(nn.Module):
    """
        Symmetric MultipleNegativesRankingLoss.
    """
    def __init__(self, name):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(MNSRankingLoss, self).__init__()
        self.scale = 20.0
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def cos_sim(self, r1, r2):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        r1_norm = torch.nn.functional.normalize(r1, p=2, dim=1)
        r2_norm = torch.nn.functional.normalize(r2, p=2, dim=1)
        return torch.mm(r1_norm, r2_norm.transpose(0, 1))


    def forward(self, r1, r2):
        scores = self.cos_sim(r1, r2) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]

        anchor_positive_scores = scores[:, 0:len(r2)]
        forward_loss = self.cross_entropy_loss(scores, labels)
        backward_loss = self.cross_entropy_loss(anchor_positive_scores.transpose(0, 1), labels)
        return (forward_loss + backward_loss) / 2
