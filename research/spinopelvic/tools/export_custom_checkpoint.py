#!/usr/bin/env python3
"""Export a selected fine-tuned checkpoint to ONNX with numerical parity checks.

The original backend/weights files and bundled ONNX graphs are never overwritten.
Use a secure output folder outside the repository. Research ONNX artifacts should
not be published with patient-linked metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(kind: str, checkpoint_path: Path, output_dir: Path, opset: int = 17) -> dict:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from backend.models import models as model_constants
    from backend.models.training import build_s1_model, build_unet
    from backend.models.hrnet import build_hrnet_model, decode_heatmaps

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Checkpoint must contain a tensor state dictionary under 'model'")
    if kind == "vertebra":
        model = build_unet(checkpoint, classes=6)
        size, channels = model_constants.MODEL_IMAGE_SIZE, 1
    elif kind == "femoral":
        model = build_unet(checkpoint, classes=1)
        size, channels = model_constants.FEMORAL_IMAGE_SIZE, 1
    elif kind == "hrnet":
        model = build_hrnet_model(checkpoint)
        size, channels = model_constants.MODEL_IMAGE_SIZE, 1
    else:
        model = build_s1_model(int(checkpoint.get("size", model_constants.MODEL_IMAGE_SIZE)))
        size, channels = int(checkpoint.get("size", model_constants.MODEL_IMAGE_SIZE)), 3
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    if kind == "s1":
        model.roi_heads.detections_per_img = 1

    class Detector(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, image):
            output = self.network([image[0]])[0]
            return output["scores"], output["keypoints"]

    class Landmarks(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, image):
            return decode_heatmaps(self.network(image), self.network.heatmap_stride)

    network = Detector(model) if kind == "s1" else Landmarks(model) if kind == "hrnet" else model
    sample = torch.rand(1, channels, size, size)
    names = ["scores", "keypoints"] if kind == "s1" else ["output"]
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{kind}.onnx"
    torch.manual_seed(123)
    with torch.inference_mode():
        torch.onnx.export(network, (sample,), str(destination), dynamo=False, opset_version=opset,
                          input_names=["image"], output_names=names,
                          dynamic_axes={name: {0: "detections"} for name in names} if kind == "s1" else None)
    onnx.checker.check_model(str(destination))
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 2
    session = ort.InferenceSession(str(destination), sess_options=settings, providers=["CPUExecutionProvider"])
    checks = []
    for tensor in (sample, torch.zeros_like(sample)):
        with torch.inference_mode():
            expected = network(tensor)
        expected = expected if isinstance(expected, tuple) else (expected,)
        actual = session.run(None, {"image": tensor.numpy()})
        for left, right in zip(expected, actual):
            np.testing.assert_allclose(left.numpy(), right, rtol=2e-3, atol=2e-3)
        checks.append("random_input" if bool(tensor.any()) else "zero_input")
    metadata = {
        "kind": kind,
        "checkpoint_path_label": checkpoint_path.name,
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "onnx_sha256": sha256(destination),
        "onnx_bytes": destination.stat().st_size,
        "opset": opset,
        "input_shape": [1, channels, size, size],
        "precision": "float32",
        "strict_state_dict_load": True,
        "onnx_checker": "passed",
        "onnxruntime_parity_inputs": checks,
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "source_repository": "frshtagl/Spine-Contour",
        "note": "Synthetic-input graph parity only; not a clinical validation or real-image parity claim.",
    }
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["vertebra", "femoral", "hrnet", "s1"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing ONNX export")
    parser.add_argument("--allow-in-repo", action="store_true", help="Only for synthetic tests; never use with real model artifacts")
    args = parser.parse_args()
    try:
        if not args.allow_in_repo:
            out = args.output.expanduser().resolve()
            try:
                out.relative_to(ROOT.resolve())
            except ValueError:
                pass
            else:
                raise ValueError(f"Refusing to write ONNX model inside Git repository: {out}")
        if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Export directory is not empty: {args.output}; use a new directory or --overwrite after recording why")
        report = export(args.kind, args.checkpoint, args.output, args.opset)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
