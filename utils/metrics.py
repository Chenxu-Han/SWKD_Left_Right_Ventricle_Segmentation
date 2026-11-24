import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def segmentation_metrics(prediction, reference):
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    count = prediction.sum() + reference.sum()
    dice = (2.0 * (prediction & reference).sum() + 1e-8) / (count + 1e-8)
    if not prediction.any() and not reference.any():
        return dice, 0.0
    if not prediction.any() or not reference.any():
        return dice, float(np.hypot(*prediction.shape))
    pred_surface = prediction ^ binary_erosion(prediction)
    ref_surface = reference ^ binary_erosion(reference)
    distances = np.concatenate((distance_transform_edt(~ref_surface)[pred_surface], distance_transform_edt(~pred_surface)[ref_surface],))
    return dice, float(np.percentile(distances, 95))
