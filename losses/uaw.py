"""Uncertainty-aware weighting (UAW), Eqs. (8)-(10)."""

import math
import torch
from torch import nn
from torch.nn import functional as F
import config


class UAWLoss(nn.Module):
    def forward(self, student_log_probability, teacher_logits, target):
        """Eqs. (8)-(10): label correction and entropy-weighted KL divergence."""
        with torch.no_grad():
            probability = teacher_logits.softmax(dim=1)
            log_probability = teacher_logits.log_softmax(dim=1)
            entropy = -(probability * log_probability).sum(1) / math.log(teacher_logits.shape[1])
            weight = (entropy + config.EPSILON).pow(config.GAMMA)
            weight = weight / weight.mean((-2, -1), keepdim=True)
            one_hot = F.one_hot(target, teacher_logits.shape[1]).permute(0, 3, 1, 2).bool()
            competitor = teacher_logits.masked_fill(one_hot, -torch.inf).amax(1, keepdim=True)
            annotated = torch.maximum(teacher_logits.gather(1, target[:, None]), competitor + config.MARGIN)
            corrected = teacher_logits.scatter(1, target[:, None], annotated)
            soft_probability = (corrected / config.TEMPERATURE).softmax(1)
            soft_log_probability = (corrected / config.TEMPERATURE).log_softmax(1)
        kl = (soft_probability * (soft_log_probability - student_log_probability)).sum(1)
        return config.TEMPERATURE ** 2 * (weight * kl).mean()
