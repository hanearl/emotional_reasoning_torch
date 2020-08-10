import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCELoss()

    def forward(self, y_pred, y_true):
        bce = self.bce_loss(y_pred, y_true)
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        gamma_factor = (1 - y_pred) ** self.gamma
        focal_loss = alpha_factor * gamma_factor * bce

        return torch.mean(focal_loss)