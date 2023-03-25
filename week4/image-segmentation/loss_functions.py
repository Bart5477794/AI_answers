from torch import nn
import torch


def dice_loss(pred, target, smooth=0):
    """
    TODO:
    Part 2: 
        - Calculate dice loss as per lab instructions
    """

    loss = 1 - 2 * torch.sum(pred * target) / (torch.sum(pred**2) + torch.sum(target**2))
    return loss


def mixed_loss(pred, target):
    """
    TODO:
    Part 2: 
        - Calculate mixed loss as per lab instructions
    """
    loss_bce = nn.BCELoss()
    loss = 0.5 * dice_loss(pred, target) + 0.5 * loss_bce(pred, target)
    return loss
