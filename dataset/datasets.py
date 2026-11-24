from array import array
from collections import Counter
from pathlib import Path
import re
import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset
import config


def read_record(line, data_root):
    fields = line.strip().split("--------")
    if len(fields) not in (5, 6) or fields[0] != "LVRV":
        raise ValueError("Expected LVRV, dataset, image, LV mask, RV mask, and an optional view field.")
    tokens = re.split(r"[/\\_.\s-]+", fields[2].lower())
    views = {"a2c": "a2c", "2ch": "a2c", "a3c": "a3c", "3ch": "a3c", "a4c": "a4c", "4ch": "a4c"}
    view = (fields[5].lower() if len(fields) == 6 else next((views[t] for t in tokens if t in views), "unknown"))
    view = views.get(view, view)
    paths = [None if value == "None" else data_root / Path(value) for value in fields[2:5]]
    return fields[1], paths, view


class EchoDataset(Dataset):
    def __init__(self, manifest, teacher_input=False, data_root=None, datasets=config.DATASETS, original_resolution=False, ):
        self.manifest = config.resolve_path(manifest)
        self.data_root = config.resolve_path(data_root) if data_root is not None else self.manifest.parent
        self.teacher_input = teacher_input
        self.original_resolution = original_resolution
        self.offsets = array("Q")
        self.excluded = Counter()
        with self.manifest.open("rb") as file:
            offset = 0
            for line in file:
                if line.strip():
                    dataset_name, paths, view = read_record(line.decode("utf-8-sig"), self.data_root)
                    if datasets is not None and dataset_name not in datasets:
                        self.excluded["dataset"] += 1
                    elif (paths[0] is None or paths[1] is None or (paths[2] is None and view not in ("a2c", "a3c"))):
                        self.excluded["missing_annotation"] += 1
                    elif any(not path_name.is_file() for path_name in paths if path_name is not None):
                        self.excluded["missing_file"] += 1
                    else:
                        self.offsets.append(offset)
                offset += len(line)
        print(f"{self.manifest.name}: {len(self.offsets)} included; excluded: {dict(self.excluded)}")
        if not self.offsets:
            raise ValueError(f"No eligible samples in {self.manifest}. Check paths and view-specific annotations.")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.manifest.open("rb") as file:
            file.seek(self.offsets[index])
            dataset_name, paths, view = read_record(file.readline().decode("utf-8-sig"), self.data_root)
        with PILImage.open(paths[0]) as file:
            gray = file.convert("L")
            size = gray.size if self.original_resolution else (config.IMAGE_SIZE, config.IMAGE_SIZE)
            pixels = (np.asarray(gray.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), PILImage.Resampling.BILINEAR), dtype=np.float32, ) / 255.0)
            teacher_image = (np.asarray(gray.resize((1024, 1024), PILImage.Resampling.BILINEAR), dtype=np.float32) / 255.0 if self.teacher_input else np.empty((0,), dtype=np.float32))
        masks = torch.zeros((2, size[1], size[0]), dtype=torch.bool)
        for organ in range(2):
            if paths[organ + 1] is not None:
                with PILImage.open(paths[organ + 1]) as file:
                    mask = np.asarray(file.convert("L").resize(size, PILImage.Resampling.NEAREST))
                masks[organ] = torch.from_numpy(mask > 0)
        target = torch.zeros((size[1], size[0]), dtype=torch.long)
        target[masks[0]] = 1
        target[masks[1]] = 2
        masks = torch.stack([target == 1, target == 2])
        return {"image": torch.from_numpy(pixels[None]), "teacher_image": torch.from_numpy(
            teacher_image[None]), "masks": masks, "target": target, "available": torch.ones(2, dtype=torch.bool), "dataset": dataset_name, "view": view, "path": str(paths[0]), }
