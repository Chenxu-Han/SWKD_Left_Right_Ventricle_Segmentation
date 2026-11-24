import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image as PILImage
from torch.nn import functional as F
from tqdm import tqdm
import config
from utils.checkpoint import load_student
from dataset.datasets import read_record

parser = argparse.ArgumentParser(description="Prompt-free student inference; output labels: background=0, LV=1, RV=2.")
parser.add_argument("--weights", type=Path, required=True)
parser.add_argument("--student", choices=["DeepLabV3_MobileNetV2", "Mobile_UNet", "FssNet"], default=None)
parser.add_argument("--data-root", type=Path, default=None)
parser.add_argument("--input", type=Path, default=config.DATA_DIR / "LVRV-test.txt")
parser.add_argument("--output", type=Path, default=config.ROOT / "predictions")
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main(args):
    for key in ("weights", "input", "output", "data_root"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, config.resolve_path(value))
    model = load_student(args.weights, args.student).to(args.device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.input.suffix.lower() == ".txt":

        def image_paths():
            with args.input.open(encoding="utf-8-sig") as file:
                for line in file:
                    if line.strip():
                        _, paths, _ = read_record(line, args.data_root or args.input.parent)
                        yield paths[0]

        paths = image_paths()
    elif args.input.is_dir():
        paths = (file for file in sorted(args.input.rglob("*")) if file.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    else:
        paths = [args.input]
    with (args.output / "files.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["source_image", "label_mask", "lv_mask", "rv_mask"])
        for index, path_name in enumerate(tqdm(paths, desc="Inference")):
            with PILImage.open(path_name) as image_file:
                gray = image_file.convert("L")
                width, height = gray.size
                pixels = (np.asarray(gray.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), PILImage.Resampling.BILINEAR), dtype=np.float32, ) / 255)
            probability = model(torch.from_numpy(pixels[None, None]).to(args.device))
            probability = F.interpolate(probability, size=(height, width), mode="bilinear", align_corners=False)
            labels = probability.argmax(1)[0].cpu().numpy().astype(np.uint8)
            name = f"{index:07d}_{path_name.stem}"
            names = [name + suffix for suffix in ("_labels.png", "_LV.png", "_RV.png")]
            for name, pixels in zip(names, [labels, (labels == 1).astype(np.uint8) * 255, (labels == 2).astype(np.uint8) * 255]):
                PILImage.fromarray(pixels).save(args.output / name)
            writer.writerow([str(path_name), *names])


if __name__ == "__main__":
    main(parser.parse_args())
