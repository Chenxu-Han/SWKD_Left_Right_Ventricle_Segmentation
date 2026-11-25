import argparse
import hashlib
import json
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from utils.checkpoint import save_checkpoint, restore_checkpoint
from utils.train_utils import set_seed, save_preview, write_log
from dataset.datasets import EchoDataset
from losses.frd import FRDLoss
from losses.uaw import UAWLoss
from losses.segmentation import segmentation_loss
from models.model_zoo import build_student
from models.sam import SAMTeacher

parser = argparse.ArgumentParser(description="SWKD: representation learning followed by decoder-only task learning.")
parser.add_argument("--student", default="DeepLabV3_MobileNetV2", choices=["DeepLabV3_MobileNetV2", "Mobile_UNet", "FssNet"])
parser.add_argument("--train-list", type=Path, default=config.DATA_DIR / "LVRV-train.txt")
parser.add_argument("--val-list", type=Path, default=config.DATA_DIR / "LVRV-val.txt")
parser.add_argument("--teacher", type=Path, default=config.ROOT / "weights" / "sam_vit_b_01ec64.pth", help="SAM ViT-B checkpoint; use task-adapted weights to reproduce the paper.", )
parser.add_argument("--data-root", type=Path, default=None, help="Root for relative image and mask paths; defaults to the manifest directory.", )
parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, choices=[2, 4, 8, 16, 32])
parser.add_argument("--epochs", type=int, default=config.EPOCHS)
parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
parser.add_argument("--seed", type=int, default=config.SEED)
parser.add_argument("--workers", type=int, default=0)
parser.add_argument("--run-dir", type=Path, default=None)
parser.add_argument("--resume", type=Path, default=None, help="Explicit stage1/last.pth or stage2/last.pth checkpoint.")
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def validate(student, loader, device, preview_path=None, seed=42):
    student.eval()
    total = torch.zeros(2, dtype=torch.float64, device=device)
    count = torch.zeros_like(total)
    rng = random.Random(seed)
    preview_batch = rng.randrange(len(loader))
    for batch_index, batch in enumerate(tqdm(loader, desc="Validation", leave=False)):
        images = batch["image"].to(device)
        prediction = student(images).argmax(1)
        masks = batch["masks"].to(device)
        available = batch["available"].to(device)
        for organ in range(2):
            use = available[:, organ] & masks[:, organ].flatten(1).any(1)
            pred = prediction[use] == organ + 1
            reference = masks[use, organ]
            intersection = (pred & reference).sum((-2, -1)).double()
            denominator = pred.sum((-2, -1)) + reference.sum((-2, -1))
            dice = torch.where(denominator > 0, 2 * intersection / denominator.clamp_min(1), 1.0)
            total[organ] += dice.sum()
            count[organ] += use.sum()
        if preview_path and batch_index == preview_batch:
            index = rng.randrange(len(images))
            save_preview(images[index], batch["target"][index], prediction[index], preview_path)
    values = total / count.clamp_min(1)
    if not (count > 0).any():
        raise ValueError("Validation has no annotated foreground classes.")
    values[count == 0] = float("nan")
    return {"LV_DSC": values[0].item(), "RV_DSC": values[1].item(), "Mean_DSC": values[count > 0].mean().item(), }


def train_stage(stage, student, teacher, train_data, val_loader, args, run_dir):
    folder = run_dir / f"stage{stage}"
    folder.mkdir(parents=True, exist_ok=True)
    if stage == 2:
        (folder / "previews").mkdir(exist_ok=True)
    device = torch.device(args.device)
    frd = FRDLoss(student.feature_channels).to(device) if stage == 1 else None
    uaw = UAWLoss().to(device) if stage == 2 else None
    if stage == 1:
        # Eq. (3): optimize the encoder and feature alignment only.
        student.set_stage_1(True)
        parameters = student.encoder_parameters() + list(frd.parameters())
    else:
        # Eq. (4): optimize the decoder with a fixed encoder and BN statistics.
        student.set_stage_2(True)
        parameters = student.decoder_parameters()
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run_config.update({"stage": stage, "image_size": config.IMAGE_SIZE, "alpha": config.ALPHA, "gamma": config.GAMMA, "temperature": config.TEMPERATURE, "margin": config.MARGIN, "uaw_weight": config.UAW_WEIGHT, "dice_weight": config.DICE_WEIGHT, "patience": config.PATIENCE, })
    run_config["uaw_reduction"] = "batch_mean"
    run_config["checkpoint_selection"] = "last_epoch" if stage == 1 else "validation_dsc"
    state = {"epoch": 0, "step": 0, "finished": False}
    if stage == 2:
        state.update({"best": -1.0, "bad_epochs": 0})
    if args.resume is not None and args.resume.parent == folder:
        previous = torch.load(args.resume, map_location="cpu", weights_only=False)["config"]
        previous = {key.lower(): value for key, value in previous.items()}
        if stage == 1 and previous.get("checkpoint_selection") != "last_epoch":
            raise ValueError("This Stage 1 checkpoint used DSC selection; start a new run with fixed-epoch feature training.")
        if stage == 2 and previous.get("uaw_reduction") != "batch_mean":
            raise ValueError("This checkpoint averages UAW over eligible samples; resume it with the original training code.")
        for key in ("student", "train_list", "val_list", "teacher", "data_root", "batch_size", "seed", "learning_rate", "image_size", "alpha", "gamma", "temperature", "margin", "epochs", "patience", "dice_weight", "uaw_weight", "inputs",):
            if previous.get(key) != run_config[key]:
                raise ValueError(f"Resume configuration differs: {key}")
        state = restore_checkpoint(args.resume, student, frd, optimizer, scheduler)
        print(f"Resume stage {stage}: {state['epoch']} completed epochs")
    (folder / "config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    for epoch in range(state["epoch"], args.epochs):
        if state["finished"]:
            break
        student.set_stage_1(True) if stage == 1 else student.set_stage_2(True)
        generator = torch.Generator().manual_seed(args.seed + stage * 10000 + epoch)
        order = torch.randperm(len(train_data), generator=generator).tolist()
        loader = DataLoader(train_data, batch_size=args.batch_size, sampler=order, num_workers=args.workers, drop_last=True, pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(args.seed + epoch), )
        progress = tqdm(loader, desc=f"Stage {stage} | epoch {epoch + 1}/{args.epochs}")
        loss_sum, seen = 0.0, 0
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if stage == 1:
                with torch.no_grad():
                    reference = torch.cat([teacher.encode(image[None].to(device)) for image in batch["teacher_image"]])
                feature = student.encode(student.prepare_input(frd.perturb(images)))["final_feature"]
                loss = frd(feature, reference)
                distillation = loss.detach()
            else:
                target = batch["target"].to(device)
                masks = batch["masks"].to(device)
                available = batch["available"].to(device)
                output = student.forward_all(images, config.TEMPERATURE)
                loss = segmentation_loss(output, target, masks)
                eligible = available.all(1) & masks.flatten(2).any(2).all(1)
                distillation = loss * 0
                if eligible.any():
                    with torch.no_grad():
                        results = [teacher(batch["teacher_image"][index: index + 1].to(device), masks[index: index + 1]) for index in eligible.nonzero().flatten().tolist()]
                        teacher_logits = torch.cat([result["logits"] for result in results])
                    distillation = uaw(output["log_soft_probability"][eligible], teacher_logits, target[eligible])
                    # Eq. (4): unavailable teacher targets contribute zero to the batch mean.
                    distillation = distillation * eligible.float().mean()
                loss = loss + config.UAW_WEIGHT * distillation
            loss.backward()
            optimizer.step()
            state["step"] += 1
            loss_sum += loss.item() * len(images)
            seen += len(images)
            progress.set_postfix(loss=f"{loss.item():.5f}", distill=f"{distillation.item():.5f}")
        metrics = {}
        improved = False
        if stage == 2:
            metrics = validate(student, val_loader, device, folder / "previews" / f"epoch_{epoch + 1:03d}.png", args.seed + epoch)
            improved = metrics["Mean_DSC"] > state["best"]
            state["best"] = max(state["best"], metrics["Mean_DSC"])
            state["bad_epochs"] = 0 if improved else state["bad_epochs"] + 1
        write_log(folder / "epochs.csv", {"Epoch": epoch + 1, "Loss": loss_sum / seen, "LR": optimizer.param_groups[0]["lr"], **metrics}, )
        scheduler.step()
        state.update({"epoch": epoch + 1, "finished": epoch + 1 == args.epochs or (stage == 2 and state["bad_epochs"] >= config.PATIENCE), })
        if improved:
            save_checkpoint(folder / "best.pth", student, frd, optimizer, scheduler, state, run_config)
        save_checkpoint(folder / "last.pth", student, frd, optimizer, scheduler, state, run_config)
        print(f"Stage {stage}, epoch {epoch + 1}: loss={loss_sum / seen:.5f}" + (f", {metrics}" if metrics else ""))
    selected = torch.load(folder / ("last.pth" if stage == 1 else "best.pth"), map_location="cpu", weights_only=False)
    student.load_state_dict(selected["model_state_dict"], strict=True)
    return selected


def main(args):
    set_seed(args.seed)
    for key in ("train_list", "val_list", "teacher", "data_root", "run_dir", "resume"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, config.resolve_path(value))
    run_dir = args.run_dir or (args.resume.parent.parent if args.resume else config.ROOT / "runs" / f"SWKD_{args.student}_seed{args.seed}")
    start_stage = 1
    if args.resume:
        start_stage = torch.load(args.resume, map_location="cpu", weights_only=False)["config"]
        start_stage = start_stage.get("stage", start_stage.get("Stage"))
        if start_stage not in (1, 2) or args.resume.parent != run_dir / f"stage{start_stage}":
            raise ValueError("Resume must be a checkpoint inside the selected run and stage directory.")
        if args.resume.name != "last.pth":
            raise ValueError("Resume requires the last.pth checkpoint from the selected stage.")
        if start_stage == 2 and not (args.resume.parent / "best.pth").is_file():
            raise ValueError("Stage 2 resume also requires best.pth from the same run.")
        if start_stage == 1 and (run_dir / "stage2" / "last.pth").exists():
            raise ValueError("Stage 2 has already started; resume its last.pth checkpoint.")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("Output directory is not empty. Use --resume or choose a new --run-dir.")
    if args.epochs < 1 or args.workers < 0 or args.learning_rate <= 0:
        raise ValueError("Epochs and learning rate must be positive; Workers must be nonnegative.")
    train_data = EchoDataset(args.train_list, teacher_input=True, data_root=args.data_root)
    val_data = EchoDataset(args.val_list, data_root=args.data_root)
    if len(train_data) < args.batch_size:
        raise ValueError("The training split must contain at least one full batch.")
    print(f"Training images: {len(train_data)}; validation images: {len(val_data)}; output: {run_dir}")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers, generator=torch.Generator().manual_seed(args.seed), )
    student = build_student(args.student).to(args.device)
    teacher = SAMTeacher(args.teacher).to(args.device).eval()
    provenance = {}
    for key, file_path in [("train", args.train_list), ("validation", args.val_list), ("teacher", args.teacher), ]:
        hash = hashlib.sha256()
        with file_path.open("rb") as file:
            for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
                hash.update(block)
        provenance[key] = {"path": str(file_path), "sha256": hash.hexdigest()}
    args.inputs = provenance
    args.inputs["included"] = {"train": len(train_data), "validation": len(val_data)}
    args.inputs["excluded"] = {"train": dict(train_data.excluded), "validation": dict(val_data.excluded)}
    if start_stage == 1:
        train_stage(1, student, teacher, train_data, None, args, run_dir)
    train_stage(2, student, teacher, train_data, val_loader, args, run_dir)
    print(f"Inference weights: {run_dir / 'stage2' / 'best.pth'}")


if __name__ == "__main__":
    main(parser.parse_args())
