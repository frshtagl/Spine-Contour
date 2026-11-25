#!/usr/bin/env python3
"""Run local femoral U-Net inference and export original crop-frame masks.

Input images should be the S1-framed crops produced by make_source_frames.py.
The preprocessing/threshold/flip-TTA match the parent app's femoral inference path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from train_femoral_unet import (
    SIZE_CANVAS, SIZE_MODEL, THRESHOLD, checkpoint_state, make_model,
    preprocess, read_gray, sha256_file,
)

SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ROOT = Path(__file__).resolve().parents[3]


def restore_mask(probability: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    height, width = original_shape
    scale = min(SIZE_CANVAS / height, SIZE_CANVAS / width)
    resized_height = max(1, round(height * scale))
    resized_width = max(1, round(width * scale))
    top = (SIZE_CANVAS - resized_height) // 2
    left = (SIZE_CANVAS - resized_width) // 2
    full_canvas = cv2.resize(probability.astype(np.float32), (SIZE_CANVAS, SIZE_CANVAS), interpolation=cv2.INTER_LINEAR)
    crop = full_canvas[top:top + resized_height, left:left + resized_width]
    restored = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    return (restored >= THRESHOLD).astype(np.uint8) * 255


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="S1-framed grayscale image folder")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", default=None)
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--out", type=Path, required=True, help="Secure output folder outside Git")
    parser.add_argument("--no-tta-flip", action="store_true")
    parser.add_argument("--allow-in-repo", action="store_true", help="Only for synthetic tests; never use with real data")
    args = parser.parse_args()
    try:
        if not args.allow_in_repo:
            try:
                args.out.expanduser().resolve().relative_to(ROOT.resolve())
            except ValueError:
                pass
            else:
                raise ValueError("Refusing to write case-level prediction masks inside Git")
        if args.out.exists() and any(args.out.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {args.out}")
        checkpoint_hash = sha256_file(args.checkpoint)
        if args.expected_checkpoint_sha256 and checkpoint_hash.lower() != args.expected_checkpoint_sha256.lower():
            raise ValueError(f"Checkpoint SHA-256 mismatch: expected {args.expected_checkpoint_sha256}, got {checkpoint_hash}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        state, encoder = checkpoint_state(checkpoint, args.encoder)
        model = make_model(encoder)
        model.load_state_dict(state, strict=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        files = sorted(p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
        if not files:
            raise ValueError(f"No supported images in {args.images}")
        args.out.mkdir(parents=True, exist_ok=True)
        mask_dir = args.out / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        with torch.no_grad():
            for path in files:
                image = read_gray(path)
                x, _ = preprocess(image)
                tensor = torch.from_numpy(x[None, None].astype(np.float32)).to(device)
                probability = torch.sigmoid(model(tensor))
                if not args.no_tta_flip:
                    flipped = torch.sigmoid(model(tensor.flip(-1))).flip(-1)
                    probability = (probability + flipped) / 2.0
                prob = probability[0, 0].detach().cpu().numpy()
                mask = restore_mask(prob, image.shape)
                destination = mask_dir / f"{path.stem}.png"
                Image.fromarray(mask).save(destination)
                rows.append({
                    "file": path.name,
                    "source_sha256": sha256_file(path),
                    "width_px": int(image.shape[1]),
                    "height_px": int(image.shape[0]),
                    "mask_file": destination.name,
                    "mask_sha256": sha256_file(destination),
                    "foreground_pixels": int((mask > 0).sum()),
                })
        manifest = {
            "schema_version": "spinecontour-spinopelvic.femoral-predictions/1",
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_bytes": args.checkpoint.stat().st_size,
            "encoder": encoder,
            "threshold": THRESHOLD,
            "flip_tta": not args.no_tta_flip,
            "preprocessing": "robust_rescale+CLAHE2x8x8+letterbox768+resize640+gray_norm; restore to crop frame",
            "device": device,
            "images": rows,
            "note": "Masks are in the S1-framed crop coordinate system; restore to original radiograph coordinates using frame_manifest_all.json before geometry analysis.",
        }
        (args.out / "prediction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"predicted_masks": len(rows), "checkpoint_sha256": checkpoint_hash, "output": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
