#!/usr/bin/env python3
"""Inspect the real Spine-Contour LFS checkpoints and verify architecture loading.

Run from the repository root after:
  git lfs pull
  python -m pip install -r backend/requirements-export.txt

This prints checkpoint metadata, tensor shapes, and strict-load results. It does not
train, export, modify, or upload any weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = ROOT / "backend" / "weights"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(kind: str) -> dict:
    import torch
    from backend.models.training import build_s1_model, build_unet
    from backend.models.hrnet import build_hrnet_model
    from backend.models.models import MODEL_IMAGE_SIZE

    paths = {
        "vertebra": WEIGHTS / "vertebra_unet.pt",
        "femoral": WEIGHTS / "femoral_unet.pt",
        "hrnet": WEIGHTS / "hrnet_landmarks.pt",
        "s1": WEIGHTS / "s1_keypointrcnn.pt",
    }
    path = paths[kind]
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint {path}; run git lfs pull first.")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"{path.name} does not have the expected checkpoint['model'] mapping.")

    if kind == "vertebra":
        model = build_unet(checkpoint, classes=6)
    elif kind == "femoral":
        model = build_unet(checkpoint, classes=1)
    elif kind == "hrnet":
        model = build_hrnet_model(checkpoint)
    else:
        model = build_s1_model(int(checkpoint.get("size", MODEL_IMAGE_SIZE)))
    model.load_state_dict(checkpoint["model"], strict=True)

    state = checkpoint["model"]
    return {
        "kind": kind,
        "file": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "strict_load": "passed",
        "metadata_keys": sorted(k for k in checkpoint.keys() if k != "model"),
        "metadata": {
            k: checkpoint[k]
            for k in ("encoder", "backbone", "stride", "size", "landmarks", "epoch", "best_metric")
            if k in checkpoint and isinstance(checkpoint[k], (str, int, float, bool, list, tuple))
        },
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "tensor_count": len(state),
        "tensor_shapes": {k: list(v.shape) for k, v in state.items() if hasattr(v, "shape")},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["vertebra", "femoral", "hrnet", "s1", "all"], default="all")
    args = parser.parse_args()
    kinds = ["vertebra", "femoral", "hrnet", "s1"] if args.kind == "all" else [args.kind]
    results = []
    for kind in kinds:
        results.append(inspect(kind))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
