from torch.nn import functional as F
import config


def segmentation_loss(output, target, masks):
    # Complete categorical supervision; absent RV in A2C/A3C is background.
    cross_entropy = F.nll_loss(output["log_probability"], target)
    probability = output["probability"][:, 1:]
    intersection = (probability * masks).sum((-2, -1))
    denominator = probability.sum((-2, -1)) + masks.sum((-2, -1))
    dice = (2 * intersection + config.EPSILON) / (denominator + config.EPSILON)
    return cross_entropy + config.DICE_WEIGHT * (1 - dice.mean())
