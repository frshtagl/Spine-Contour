#!/usr/bin/env python3
"""Audit the Spine-Contour spinopelvic research scaffold without opening patient data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PRESENT = [
    "research/spinopelvic/README.md",
    "research/spinopelvic/STATUS.md",
    "research/spinopelvic/ANNOTATION_PROTOCOL.md",
    "research/spinopelvic/TRAINING_PLAN.md",
    "research/spinopelvic/SYNOPSIS_ALIGNMENT.md",
    "research/spinopelvic/SYNOPSIS_AMENDMENT_RECOMMENDATIONS.md",
    "research/spinopelvic/configs/project.yaml",
    "research/spinopelvic/schemas/annotation.schema.json",
    "research/spinopelvic/schemas/annotation.example.json",
    "research/spinopelvic/tests/test_annotation_schema.py",
    "research/spinopelvic/tools/validate_annotations.py",
    "research/spinopelvic/tools/inspect_checkpoints.py",
    "research/spinopelvic/docs/ANNOTATION_STRATEGY_EXACT.md",
    "research/spinopelvic/docs/REPOSITORY_CROSS_CHECK_2026-11-25.md",
]
PENDING = [
    "research/spinopelvic/tools/prepare_splits.py",
    "research/spinopelvic/tools/create_annotation_templates.py",
    "research/spinopelvic/tools/build_dataset.py",
    "research/spinopelvic/tools/train_segmentation.py",
    "research/spinopelvic/tools/train_hrnet.py",
    "research/spinopelvic/tools/train_s1.py",
    "research/spinopelvic/tools/evaluate.py",
    "research/spinopelvic/tools/export_checkpoint.py",
    "research/spinopelvic/colab/spinopelvic_training.ipynb",
]
WEIGHTS = [
    "backend/weights/vertebra_unet.pt",
    "backend/weights/femoral_unet.pt",
    "backend/weights/hrnet_landmarks.pt",
    "backend/weights/s1_keypointrcnn.pt",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    present = [path for path in PRESENT if (root / path).is_file()]
    missing_present = [path for path in PRESENT if not (root / path).is_file()]
    pending_present = [path for path in PENDING if (root / path).is_file()]
    pending_missing = [path for path in PENDING if not (root / path).is_file()]
    lfs = {}
    for relative in WEIGHTS:
        path = root / relative
        if not path.is_file():
            lfs[relative] = "missing"
            continue
        with path.open("rb") as handle:
            prefix = handle.read(200)
        lfs[relative] = "LFS_POINTER" if prefix.startswith(b"version https://git-lfs.github.com/spec/v1") else "file_present_tensor_load_not_tested"
    report = {
        "repository_root": str(root),
        "present_research_files": present,
        "missing_expected_research_files": missing_present,
        "pipeline_components_present": pending_present,
        "pipeline_components_still_missing": pending_missing,
        "checkpoint_file_status": lfs,
        "scaffold_complete": not missing_present and not pending_missing,
        "note": "File presence is not proof of strict checkpoint loading, training correctness or clinical validity.",
    }
    print(json.dumps(report, indent=2))
    return 0 if report["scaffold_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
