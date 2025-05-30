import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_one_hot(targets, n_classes, smoothing=0.0, device='cpu'):
    """
    Apply label smoothing to one-hot targets.

    Args:
        targets: [B] tensor of target labels
        n_classes: number of classes
        smoothing: smoothing factor (0 means no smoothing)
        device: device for output tensor

    Returns:
        smoothed one-hot tensor of shape [B, n_classes]
    """
    assert 0 <= smoothing < 1
    with torch.no_grad():
        confidence = 1.0 - smoothing
        label_shape = torch.Size((targets.size(0), n_classes))
        smooth_targets = torch.full(label_shape, smoothing / (n_classes - 1), device=device)
        smooth_targets.scatter_(1, targets.unsqueeze(1), confidence)
    return smooth_targets


class GCELoss(nn.Module):
    def __init__(self, q=0.5, label_smoothing=0.0):
        super().__init__()
        self.q = q
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        n_classes = logits.size(1)
        device = logits.device

        if self.label_smoothing > 0:
            targets_smoothed = smooth_one_hot(targets, n_classes, self.label_smoothing, device)
            p_t = torch.sum(probs * targets_smoothed, dim=1)
        else:
            p_t = probs[torch.arange(logits.size(0)), targets]

        loss = (1 - p_t ** self.q) / self.q
        return loss.mean()


class SCELoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        n_classes = logits.size(1)
        device = logits.device

        # Standard CE with label smoothing if enabled
        if self.label_smoothing > 0:
            targets_smoothed = smooth_one_hot(targets, n_classes, self.label_smoothing, device)
            ce = (-targets_smoothed * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        else:
            ce = F.cross_entropy(logits, targets)

        # Reverse CE term
        if self.label_smoothing > 0:
            rce = (-torch.sum(probs * torch.log(targets_smoothed + 1e-7), dim=1)).mean()
        else:
            one_hot = F.one_hot(targets, num_classes=n_classes).float()
            rce = (-torch.sum(probs * torch.log(one_hot + 1e-7), dim=1)).mean()

        return self.alpha * ce + self.beta * rce


class BootstrappingLoss(nn.Module):
    def __init__(self, beta=0.95, label_smoothing=0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        n_classes = logits.size(1)
        device = logits.device

        if self.label_smoothing > 0:
            true_labels = smooth_one_hot(targets, n_classes, self.label_smoothing, device)
        else:
            true_labels = F.one_hot(targets, num_classes=n_classes).float()

        blended_labels = self.beta * true_labels + (1 - self.beta) * probs.detach()
        loss = -torch.sum(blended_labels * F.log_softmax(logits, dim=1), dim=1)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        n_classes = logits.size(1)
        device = logits.device

        if self.label_smoothing > 0:
            targets_smoothed = smooth_one_hot(targets, n_classes, self.label_smoothing, device)
            # Compute p_t = sum of probs weighted by smoothed targets (soft labels)
            p_t = torch.sum(probs * targets_smoothed, dim=1)
            # Compute focal weight for each example
            focal_weight = (1 - p_t) ** self.gamma
            # Compute loss per sample as sum over classes of (targets_smoothed * log_probs)
            loss = -torch.sum(targets_smoothed * log_probs, dim=1)
            loss = focal_weight * loss
        else:
            p_t = probs[torch.arange(logits.size(0)), targets]
            focal_weight = (1 - p_t) ** self.gamma
            loss = -focal_weight * log_probs[torch.arange(logits.size(0)), targets]

        return loss.mean()


class MAELoss(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        n_classes = logits.size(1)
        device = logits.device

        if self.label_smoothing > 0:
            one_hot = smooth_one_hot(targets, n_classes, self.label_smoothing, device)
        else:
            one_hot = F.one_hot(targets, num_classes=n_classes).float()

        return torch.abs(probs - one_hot).mean()


class ELRLoss(nn.Module):
    def __init__(self, teacher_model, lambda_elr=3.0, device='cuda', criterion=None, label_smoothing=0.0):
        super().__init__()
        self.teacher_model = teacher_model
        self.lambda_elr = lambda_elr
        self.device = device
        self.label_smoothing = label_smoothing
        self.criterion = criterion or nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, student_logits, targets, inputs):
        with torch.no_grad():
            teacher_logits = self.teacher_model(inputs)
            teacher_probs = F.softmax(teacher_logits, dim=1)

        student_probs = F.softmax(student_logits, dim=1)

        # Use smoothed labels if label_smoothing > 0
        if self.label_smoothing > 0:
            n_classes = student_logits.size(1)
            device = student_logits.device
            targets_smoothed = smooth_one_hot(targets, n_classes, self.label_smoothing, device)
            # CrossEntropyLoss expects integer targets, so we compute CE manually:
            ce_loss = (-targets_smoothed * F.log_softmax(student_logits, dim=1)).sum(dim=1).mean()
        else:
            ce_loss = self.criterion(student_logits, targets)

        elr_reg = -torch.sum(student_probs * torch.log(teacher_probs + 1e-6), dim=1).mean()

        return ce_loss + self.lambda_elr * elr_reg

class LossManager:
    """Manages loss functions"""
    @staticmethod
    def get_loss(loss_name: str, label_smoothing=0.0, teacher_model=None, base_criterion_for_elr=None, config=None):
        loss_lower = loss_name.lower()
        if loss_lower == 'ce':
            return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        elif loss_lower == 'gce':
            return GCELoss(q=0.3, label_smoothing=label_smoothing) # Assuming q is hardcoded or from config
        elif loss_lower == 'sce':
            return SCELoss(alpha=1.0, beta=1.0, label_smoothing=label_smoothing) # Assuming alpha, beta are hardcoded or from config
        elif loss_lower == 'bootstrapping':
            return BootstrappingLoss(beta=0.95, label_smoothing=label_smoothing) # Assuming beta is hardcoded or from config
        elif loss_lower == 'focal':
            return FocalLoss(gamma=2.0, label_smoothing=label_smoothing) # Assuming gamma is hardcoded or from config
        elif loss_lower == 'mae':
            return MAELoss(label_smoothing=label_smoothing) # Ensure this is appropriate for your task
        elif loss_lower == 'elr':
            if teacher_model is None or base_criterion_for_elr is None or config is None:
                raise ValueError("ELRLoss requires teacher_model, base_criterion_for_elr, and config.")
            return ELRLoss(teacher_model=teacher_model,
                           criterion=base_criterion_for_elr)
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")