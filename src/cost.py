import torch.nn as nn
from torch.nn import TripletMarginLoss


class TripletLoss(nn.Module):
    """Triplet Loss definition"""

    def __init__(self, device: int = 0):
        super(TripletLoss, self).__init__()
        self.device = device
        self.triplet_loss = TripletMarginLoss(margin=1, p=2, eps=1e-7)

    def forward(self, info, batch):
        """forward method"""

        anchor = info[0].to(self.device)
        positive = info[1].to(self.device)
        negative = info[2].to(self.device)
        loss = self.triplet_loss(anchor, positive, negative)
        return loss


# TODO : Classfication loss
