#!/usr/bin/env python3
"""Audit the Spine-Contour spinopelvic research workspace without opening patient data."""
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
    "research/spinopelvic/configs/experiment_template.yaml",
    "research/spinopelvic/configs/checkpoint_manifest.example.json",
    "research/spinopelvic/schemas/annotation.schema.json",
    "research/spinopelvic/schemas/annotation.example.json",
    "research/spinopelvic/schemas/split_manifest.schema.json",
    "research/spinopelvic/schemas/reference_measurements.schema.json",
    "research/spinopelvic/schemas/reference_measurements.example.json",
    "research/spinopelvic/tests/test_annotation_schema.py",
    "research/spinopelvic/tests/test_research_pipeline.py",
    "research/spinopelvic/tests/test_reference_measurements_schema.py",
    "research/spinopelvic/tests/test_prediction_normalizer.py",
    "research/spinopelvic/tools/validate_annotations.py",
    "research/spinopelvic/tools/make_annotation_templates.py",
    "research/spinopelvic/tools/prepare_splits.py",
    "research/spinopelvic/tools/build_training_inputs.py",
    "research/spinopelvic/tools/compute_measurements_from_annotations.py",
    "research/spinopelvic/tools/train_femoral_unet.py",
    "research/spinopelvic/tools/predict_femoral_unet.py",
    "research/spinopelvic/tools/make_source_frames.py",
    "research/spinopelvic/tools/export_custom_checkpoint.py",
    "research/spinopelvic/tools/normalize_model_predictions.py",
    "research/spinopelvic/tools/validate_end_to_end_framing.py",
    "research/spinopelvic/tools/validate_synthetic_s1_search.py",
    "research/spinopelvic/tools/import_source_bundle.sh",
    "research/spinopelvic/tools/inspect_checkpoints.py",
    "research/spinopelvic/analysis/analyze_agreement.py",
    "research/spinopelvic/analysis/paired_predictions_template.csv",
    "research/spinopelvic/requirements-research.txt",
    "research/spinopelvic/requirements-research-test.txt",
    "research/spinopelvic/docs/ANNOTATION_STRATEGY_EXACT.md",
    "research/spinopelvic/docs/REPOSITORY_CROSS_CHECK_2026-11-25.md",
    "research/spinopelvic/docs/RECOVERED_KIT_REVIEW.md",
    "research/spinopelvic/docs/DATA_PIPELINE_RUNBOOK.md",
    "research/spinopelvic/docs/RESEARCH_DATA_GOVERNANCE.md",
    "research/spinopelvic/docs/RESEARCH_QC_AND_STATISTICS.md",
    "research/spinopelvic/docs/EXPERIMENT_LOG_TEMPLATE.md",
    "research/spinopelvic/docs/MODEL_CARDS.md",
    "research/spinopelvic/docs/RESEARCH_PROTOCOL_DECISIONS.md",
    ".github/workflows/spinopelvic-research-tests.yml",
]
PENDING = [
    "research/spinopelvic/recovered_kits/source_bundle/contents/vertebra_unet_finetune/train.py",
    "research/spinopelvic/recovered_kits/source_bundle/contents/hrnet_landmarks_finetune/train.py",
    "research/spinopelvic/recovered_kits/source_bundle/contents/s1_finetune/train.py",
    "research/spinopelvic/tools/validate_actual_s1_framing.py",
]
OPTIONAL = [
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
    optional_present = [path for path in OPTIONAL if (root / path).is_file()]
    optional_missing = [path for path in OPTIONAL if not (root / path).is_file()]
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
        "optional_components_present": optional_present,
        "optional_components_missing": optional_missing,
        "checkpoint_file_status": lfs,
        "scaffold_complete": not missing_present and not pending_missing,
        "note": "File presence is not proof of strict checkpoint loading, training correctness or clinical validity. A missing source_bundle/contents path means the local source-only archive has not yet been imported. The Colab notebook is optional and requires site approval before any patient-data transfer.",
    }
    print(json.dumps(report, indent=2))
    return 0 if report["scaffold_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
