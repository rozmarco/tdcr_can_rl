import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanSquaredError(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, prediction, target):
        return F.mse_loss(prediction, target, reduction=self.reduction)