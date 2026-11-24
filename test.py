import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from utils.checkpoint import load_student
from utils.metrics import segmentation_metrics
from dataset.datasets import EchoDataset

parser = argparse.ArgumentParser(description="Independent LV/RV evaluation with per-image DSC and HD95.")
parser.add_argument("--weights", type=Path, required=True)
parser.add_argument("--student", choices=["DeepLabV3_MobileNetV2", "Mobile_UNet", "FssNet"], default=None)
parser.add_argument("--test-list", type=Path, default=config.DATA_DIR / "LVRV-test.txt")
parser.add_argument("--data-root", type=Path, default=None)
parser.add_argument("--output", type=Path, default=config.ROOT / "results")
parser.add_argument("--resolution", choices=["128", "original"], default="128")
parser.add_argument("--external", action="store_true", help="Evaluate every dataset in the provided manifest.")
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main(args):
    for key in ("weights", "test_list", "data_root", "output"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, config.resolve_path(value))
    data = EchoDataset(args.test_list, data_root=args.data_root, datasets=None if args.external else config.DATASETS, original_resolution=args.resolution == "original", )
    model = load_student(args.weights, args.student).to(args.device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    scores = defaultdict(list)
    empty = defaultdict(int)
    counts = defaultdict(int)
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset", "image", "view", "organ", "DSC", "HD95_px", "reference_empty"])
        for batch in tqdm(DataLoader(data, batch_size=1), desc="Test"):
            probability = model(batch["image"].to(args.device))
            probability = F.interpolate(probability, size=batch["target"].shape[-2:], mode="bilinear", align_corners=False)
            prediction = probability.argmax(1)[0].cpu().numpy()
            dataset_name = batch["dataset"][0]
            counts[dataset_name] += 1
            counts["ALL"] += 1
            for organ, name in enumerate(("LV", "RV")):
                reference = batch["masks"][0, organ].numpy()
                if not reference.any():
                    empty[(dataset_name, name)] += 1
                    empty[("ALL", name)] += 1
                    writer.writerow([dataset_name, batch["path"][0], batch["view"][0], name, "", "", True])
                    continue
                dice, hd95 = segmentation_metrics(prediction == organ + 1, reference)
                scores[(dataset_name, name)].append((dice, hd95))
                scores[("ALL", name)].append((dice, hd95))
                writer.writerow([dataset_name, batch["path"][0], batch["view"][0], name, dice, hd95, False])
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset", "organ", "images", "evaluated", "empty_reference", "DSC_mean", "DSC_std", "HD95_px_mean", "HD95_px_std", "resolution", ])
        for dataset_name in sorted(counts):
            for organ in ("LV", "RV"):
                values = np.asarray(scores[(dataset_name, organ)], dtype=float).reshape(-1, 2)
                mean = values.mean(0).tolist() if len(values) else ["", ""]
                std = values.std(0, ddof=1).tolist() if len(values) > 1 else ["", ""]
                row = [dataset_name, organ, counts[dataset_name], len(values), empty[(dataset_name, organ)], mean[0], std[0], mean[1], std[1], args.resolution, ]
                writer.writerow(row)
                print(row)
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main(parser.parse_args())
