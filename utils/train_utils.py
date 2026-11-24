import csv
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image as PILImage


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_preview(images, target, prediction, path_name):
    colors = np.array([[0, 0, 0], [225, 80, 60], [40, 160, 220]], dtype=np.uint8)
    gray = (images[0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    gray = np.repeat(gray[..., None], 3, axis=-1)
    canvas = np.concatenate([gray, colors[target.cpu().numpy()], colors[prediction.cpu().numpy()]], axis=1)
    PILImage.fromarray(canvas).save(path_name)


def write_log(path_name, values):
    exists = Path(path_name).exists()
    with Path(path_name).open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(values))
        if not exists:
            writer.writeheader()
        writer.writerow(values)
