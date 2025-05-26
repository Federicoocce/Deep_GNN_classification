"""
robust_loss_functions.py

Collection of robust loss functions for training classifiers with noisy labels.
Each loss is implemented as a PyTorch nn.Module with documentation and comments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCELoss(nn.Module):
    """
    Generalized Cross Entropy Loss
    Combines properties of MAE and Cross Entropy. Robust to noisy labels.

    Args:
        q (float): Robustness parameter (0 < q <= 1). Default is 0.7.
    """
    def __init__(self, q=0.7):
        super().__init__()
        self.q = q

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        p_t = probs[torch.arange(logits.size(0)), targets]
        loss = (1 - p_t ** self.q) / self.q
        return loss.mean()


class SCELoss(nn.Module):
    """
    Symmetric Cross Entropy Loss
    Sum of cross entropy and reverse cross entropy.

    Args:
        alpha (float): Weight for standard cross-entropy.
        beta (float): Weight for reverse cross-entropy.
    """
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        ce = F.cross_entropy(logits, targets)

        one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        rce = (-torch.sum(probs * torch.log(one_hot + 1e-7), dim=1)).mean()

        return self.alpha * ce + self.beta * rce


class BootstrappingLoss(nn.Module):
    """
    Bootstrapping Loss
    Blends true labels and predicted probabilities to reduce effect of noisy labels.

    Args:
        beta (float): Weight for ground truth (0 to 1).
    """
    def __init__(self, beta=0.95):
        super().__init__()
        self.beta = beta

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        true_labels = F.one_hot(targets, num_classes=logits.size(1)).float()
        blended_labels = self.beta * true_labels + (1 - self.beta) * probs.detach()
        loss = -torch.sum(blended_labels * F.log_softmax(logits, dim=1), dim=1)
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss
    Originally designed for class imbalance; useful with noisy labels.

    Args:
        gamma (float): Focusing parameter (commonly 2.0).
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        p_t = probs[torch.arange(logits.size(0)), targets]
        focal_weight = (1 - p_t) ** self.gamma
        loss = -focal_weight * log_probs[torch.arange(logits.size(0)), targets]
        return loss.mean()


class MAELoss(nn.Module):
    """
    Mean Absolute Error Loss
    Very robust to noisy labels but harder to optimize.
    """
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        return torch.abs(probs - one_hot).mean()
