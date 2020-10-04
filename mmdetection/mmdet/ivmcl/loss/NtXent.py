import torch
from torch import nn


class NtXent(nn.Module):

    def __init__(self, bsz, T):
        super(NtXent, self).__init__()
        self.bsz = bsz
        self.T = T
        self.neg_mask = self._create_neg_mask()

    def _create_neg_mask(self):
        neg_mask = torch.ones((self.bsz * 2, self.bsz * 2), dtype=bool)
        neg_mask.fill_diagonal_(0)
        for i in range(self.bsz):
            neg_mask[i, self.bsz + i] = 0
            neg_mask[self.bsz + i, i] = 0

        return neg_mask

    def forward(self, z):

        sim = torch.mm(z, torch.transpose(z, 0, 1)) / self.T

        sim_i_j = torch.diag(sim, self.bsz)
        sim_j_i = torch.diag(sim, -self.bsz)
        pos = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(self.bsz * 2, 1)

        # neg_mask = torch.ones((self.bsz * 2, self.bsz * 2), dtype=bool).cuda()
        # neg_mask.fill_diagonal_(0)
        # for i in range(self.bsz):
        #     neg_mask[i, self.bsz + i] = 0
        #     neg_mask[self.bsz + i, i] = 0

        neg = sim[self.neg_mask].reshape(self.bsz * 2, -1)

        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return logits, labels


