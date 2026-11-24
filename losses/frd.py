"""Frequency reconstruction denoising (FRD), Eqs. (5)-(7), (11)."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_dct import dct_2d, idct_2d
from models.base import init_weight
import config


class FrequencyPerturbation(nn.Module):
    def __init__(self, size=config.IMAGE_SIZE, alpha=config.ALPHA):
        super().__init__()
        coordinate = torch.arange(size).float() / (size - 1)
        radius = torch.sqrt(coordinate[:, None] ** 2 + coordinate[None, :] ** 2) / math.sqrt(2)
        band = (radius * 10).long().clamp_max(9)
        masks = torch.stack([band == index for index in range(10)])
        masks[:, 0, 0] = False
        self.register_buffer("radius", radius[None, None], persistent=False)
        self.register_buffer("masks", masks[:, None], persistent=False)
        self.alpha = alpha

    def forward(self, image):
        # Eqs. (6)-(7): perturb one DCT band, excluding the DC coefficient.
        choice = torch.randint(10, (len(image),), device=image.device)
        mask = self.masks[choice].to(image.dtype)
        noise = torch.randn_like(image) / (self.radius + config.EPSILON)
        scale = ((noise.square() * mask).sum((-2, -1), keepdim=True) / mask.sum((-2, -1), keepdim=True) + config.EPSILON).sqrt()
        spectrum = dct_2d(image, norm="ortho")
        return idct_2d(spectrum + self.alpha * noise * mask / scale, norm="ortho")


class FRDLoss(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.Projection = nn.Conv2d(channels, 256, 1)
        self.apply(init_weight)
        self.perturb = FrequencyPerturbation()

    def forward(self, student_feature, teacher_feature):
        # Eq. (11): interpolate before the learnable 1x1 projection.
        feature = F.interpolate(student_feature, size=teacher_feature.shape[-2:], mode="bilinear", align_corners=False)
        return F.mse_loss(self.Projection(feature), teacher_feature)  # Eq. (5).
