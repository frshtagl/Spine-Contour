#!/usr/bin/env python3
"""Fine-tune the source one-channel ResNet-34 U-Net femoral union-mask model.

This is a new, explicit study fine-tuning recipe, not a claim to reproduce the
undisclosed original training run. Preprocessing matches backend.models.models:
robust rescale -> CLAHE(2.0, 8x8) -> 768 letterbox -> 640 resize -> ImageNet-gray
normalisation. The inference threshold remains 0.35 and flip averaging is used
for validation. All images/masks/checkpoints/logs must stay in secure local storage.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SIZE_CANVAS, SIZE_MODEL, THRESHOLD = 768, 640, 0.35
MEAN, STD = 0.449, 0.226
SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def robust_rescale(array: np.ndarray) -> np.ndarray:
    """Exact robust rescale from backend.models.models._robust_rescale."""
    image = np.asarray(array).astype(np.float32, copy=True)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Expected a non-empty grayscale image")
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)
    image[~finite] = float(np.median(image[finite]))
    foreground = image[image > 0]
    values = foreground if foreground.size >= 100 else image.reshape(-1)
    low, high = np.percentile(values, (0.5, 99.5))
    if high <= low + 1:
        return np.clip(image, 0, 255).astype(np.uint8)
    return np.clip((image - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def preprocess(image: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    image = robust_rescale(image)
    image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    h, w = image.shape
    scale = min(SIZE_CANVAS / h, SIZE_CANVAS / w)
    rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
    top, left = (SIZE_CANVAS - rh) // 2, (SIZE_CANVAS - rw) // 2
    canvas = np.zeros((SIZE_CANVAS, SIZE_CANVAS), dtype=np.uint8)
    canvas[top:top + rh, left:left + rw] = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_AREA)
    resized = cv2.resize(canvas, (SIZE_MODEL, SIZE_MODEL), interpolation=cv2.INTER_AREA)
    x = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    y = None
    if mask is not None:
        mcanvas = np.zeros((SIZE_CANVAS, SIZE_CANVAS), dtype=np.uint8)
        mcanvas[top:top + rh, left:left + rw] = cv2.resize(mask.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST)
        y = cv2.resize(mcanvas, (SIZE_MODEL, SIZE_MODEL), interpolation=cv2.INTER_NEAREST)
    return x, y


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim != 2:
        raise ValueError(f"Expected grayscale input, found {arr.shape} in {path.name}")
    return arr


class FemoralDataset(torch.utils.data.Dataset):
    def __init__(self, names: list[str], image_dir: Path, mask_dir: Path, augment: bool = False):
        self.names, self.image_dir, self.mask_dir, self.augment = names, image_dir, mask_dir, augment

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        image = read_gray(self.image_dir / name)
        mask = np.asarray(Image.open(self.mask_dir / f"{Path(name).stem}.png"), dtype=np.uint8)
        if mask.shape != image.shape:
            raise ValueError(f"Image/mask shape mismatch for {name}")
        x, y = preprocess(image, mask)
        if self.augment:
            if random.random() < 0.5:
                x, y = x[:, ::-1].copy(), y[:, ::-1].copy()
            angle = random.uniform(-5.0, 5.0)
            scale = random.uniform(0.95, 1.05)
            tx, ty = random.uniform(-0.02, 0.02) * SIZE_MODEL, random.uniform(-0.02, 0.02) * SIZE_MODEL
            matrix = cv2.getRotationMatrix2D(((SIZE_MODEL - 1) / 2, (SIZE_MODEL - 1) / 2), angle, scale)
            matrix[:, 2] += (tx, ty)
            normalised_black = (0.0 - MEAN) / STD
            x = cv2.warpAffine(x, matrix, (SIZE_MODEL, SIZE_MODEL), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=float(normalised_black))
            y = cv2.warpAffine(y, matrix, (SIZE_MODEL, SIZE_MODEL), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            x = np.clip(x * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05), -2.0, 2.0)
        return torch.from_numpy(x[None].astype(np.float32)), torch.from_numpy(y[None].astype(np.float32))


def dice_iou(probability: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    pred = probability >= THRESHOLD
    truth = target > 0.5
    union = np.logical_or(pred, truth).sum()
    dice = 1.0 if union == 0 else 2.0 * np.logical_and(pred, truth).sum() / (pred.sum() + truth.sum() + 1e-9)
    iou = 1.0 if union == 0 else np.logical_and(pred, truth).sum() / (union + 1e-9)
    return float(dice), float(iou)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum((0, 2, 3))
    denominator = probability.sum((0, 2, 3)) + target.sum((0, 2, 3))
    return 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()


@torch.no_grad()
def validate(model, loader, device: str, tta_flip: bool) -> dict:
    model.eval()
    dices, ious = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probability = torch.sigmoid(logits)
        if tta_flip:
            flipped = torch.sigmoid(model(x.flip(-1))).flip(-1)
            probability = (probability + flipped) / 2.0
        d, j = dice_iou(probability[0, 0].cpu().numpy(), y[0, 0].numpy())
        dices.append(d)
        ious.append(j)
    return {"n": len(dices), "dice": float(np.mean(dices)) if dices else None,
            "iou": float(np.mean(ious)) if ious else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--val-names", type=Path, required=True, help="Validation filenames created by build_training_inputs.py")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Existing backend/weights/femoral_unet.pt; never overwrite it")
    parser.add_argument("--out", type=Path, required=True, help="Secure output directory outside Git")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--allow-in-repo", action="store_true", help="Only for synthetic tests; never use with real data")
    args = parser.parse_args()
    try:
        if not args.allow_in_repo:
            root = Path(__file__).resolve().parents[3].resolve()
            out = args.out.expanduser().resolve()
            try:
                out.relative_to(root)
            except ValueError:
                pass
            else:
                raise ValueError(f"Refusing to write model weights/logs inside Git repository: {out}")
        if args.out.exists() and any(args.out.iterdir()):
            raise FileExistsError(f"Run directory is not empty: {args.out}; use a new run directory")
        if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
            raise ValueError("epochs, batch-size and learning rate must be positive")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        names = sorted(p.name for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES
                       and (args.masks / f"{p.stem}.png").is_file())
        val_names = {line.strip() for line in args.val_names.read_text(encoding="utf-8").splitlines() if line.strip()}
        unknown = val_names - set(names)
        if unknown:
            raise ValueError(f"Validation list contains files without masks: {sorted(unknown)[:5]}")
        train_names = [n for n in names if n not in val_names]
        if not train_names or not val_names:
            raise ValueError("Both training and validation sets must contain at least one labelled image")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        import segmentation_models_pytorch as smp
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError("Checkpoint must contain a model state dictionary")
        model = smp.Unet(encoder_name=str(checkpoint.get("encoder", "resnet34")), encoder_weights=None, in_channels=1, classes=1)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device)
        train_loader = torch.utils.data.DataLoader(FemoralDataset(train_names, args.images, args.masks, True),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
        val_loader = torch.utils.data.DataLoader(FemoralDataset(sorted(val_names), args.images, args.masks, False),
            batch_size=1, shuffle=False, num_workers=args.workers)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        args.out.mkdir(parents=True, exist_ok=True)
        baseline = validate(model, val_loader, device, True)
        log = [{"phase": "baseline", "metrics": baseline}]
        best = -1.0
        step = 0
        total_steps = max(1, args.epochs * len(train_loader))
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = []
            started = time.time()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                warm = min(1.0, (step + 1) / 30.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / total_steps))
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * warm * cosine
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y) + args.dice_weight * soft_dice_loss(logits, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                epoch_loss.append(float(loss.detach().cpu()))
                step += 1
                if args.max_steps and step >= args.max_steps:
                    break
            metrics = validate(model, val_loader, device, True)
            record = {"epoch": epoch, "train_loss": float(np.mean(epoch_loss)) if epoch_loss else None,
                      "validation": metrics, "seconds": round(time.time() - started, 1)}
            log.append(record)
            payload = {"model": model.state_dict(), "encoder": "resnet34", "classes": 1,
                       "input_size": SIZE_MODEL, "threshold": THRESHOLD, "epoch": epoch,
                       "validation": metrics, "config": {"seed": args.seed, "lr": args.lr,
                       "weight_decay": args.weight_decay, "dice_weight": args.dice_weight,
                       "preprocessing": "robust_rescale+CLAHE2x8x8+letterbox768+resize640+gray_norm"}}
            torch.save(payload, args.out / "last.pt")
            if metrics["dice"] is not None and metrics["dice"] > best:
                best = metrics["dice"]
                torch.save(payload, args.out / "best.pt")
            (args.out / "training_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(record), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        print(json.dumps({"baseline": baseline, "best_validation_dice": best, "output": str(args.out)}, indent=2))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
