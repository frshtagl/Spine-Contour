# Spine-Contour Spinopelvic Research Workspace

**Status:** scaffold created; no patient data, masks, or training runs have been added.

This workspace isolates the UHS spinopelvic research work from the existing Spine-Contour application. The branch is a complete snapshot of the current repository plus this research directory; the original application remains unchanged on `main`.

## Research target

Develop and internally test a reliability-aware framework for estimating:

- Pelvic incidence (PI)
- Pelvic tilt (PT)
- Sacral slope (SS)
- L1–S1 lumbar lordosis (LL)

from de-identified standing lateral lumbopelvic radiographs, using the uploaded tentative UHS synopsis as the study basis.

## Working sample plan

| Partition | Images | Role |
|---|---:|---|
| Training | 240 | Parameter learning |
| Validation | 60 | Model and threshold selection |
| Locked internal test | 100 | One-time final evaluation |
| **Total** | **400** | **300 development + 100 test** |

Partitions must be patient-disjoint. Prefer one examination per patient and stratify by collaborating site where feasible. The 100-image test set must not be used for training, model selection, threshold calibration, or iterative debugging.

This is the current working plan requested by the investigator. It differs from the tentative synopsis, which currently proposes approximately 420 examinations (300 development and 120 test). See [SYNOPSIS_ALIGNMENT.md](SYNOPSIS_ALIGNMENT.md).

## Two research tracks

### Model A — segmentation-to-geometry

The synopsis proposes 2D nnU-Net v2 segmentation of L1–L5 vertebral anatomy, the S1 superior-endplate region, and the femoral-head region, followed by explicit geometric extraction. The existing Spine-Contour ResNet-34 U-Net checkpoints are useful warm-start candidates, but they are **not automatically compatible with nnU-Net v2**. Keep the architecture decision explicit; do not describe a ResNet-34 U-Net experiment as nnU-Net.

### Model B — direct landmarks

The existing HRNet-W32 checkpoint predicts 22 lumbar/S1 landmarks. The six required spinopelvic points are:

1. L1 superior-endplate anterior endpoint
2. L1 superior-endplate posterior endpoint
3. S1 superior-endplate anterior endpoint
4. S1 superior-endplate posterior endpoint
5. Femoral-head centre 1
6. Femoral-head centre 2

A weight-preserving extension to 24 output channels is a candidate: retain the pretrained 22 channels and initialize two new femoral-centre channels. This requires implementation and tests; it has not yet been built.

## What is already available in this repository

The source application and its pretrained weights remain at their original paths. They are deliberately referenced rather than duplicated:

- Architectures/checkpoint builders: `backend/models/training.py`
- HRNet architecture and heatmap decoding: `backend/models/hrnet.py`
- Inference model definitions and image transforms: `backend/models/models.py`
- Lumbar/femoral framing: `backend/framing.py`
- Two-head fitting from a merged femoral mask: `backend/femoral.py`
- ONNX export: `tools/export_onnx.py`
- Existing level editor: `renderer/level-editor.js`
- Existing annotation record: `renderer/data/label-record.js`
- Annotation guide and schema: `docs/level-editor.md` and `docs/design/level-editor/label-record.schema.json`

See [BASELINE_INVENTORY.md](BASELINE_INVENTORY.md) for the pinned source revision, model inventory, LFS hashes, and compatibility gaps.

## Workspace layout

```text
research/spinopelvic/
├── README.md
├── STATUS.md
├── BASELINE_INVENTORY.md
├── SYNOPSIS_ALIGNMENT.md
├── ANNOTATION_PROTOCOL.md
├── TRAINING_PLAN.md
├── UPSTREAM_AND_LICENSE.md
├── .gitignore
├── configs/
│   └── project.yaml
├── schemas/
│   ├── annotation.schema.json
│   └── annotation.example.json
└── colab/
    └── README.md
```

## Data and security rule

The parent repository is public. **Never commit radiographs, DICOM files, hand-drawn masks, annotation JSON, patient/study manifests, reference readings, checkpoints, or derived image crops to this repository.** The nested `.gitignore` blocks the planned local data/output folders and common medical-image formats.

The synopsis states that any movement of data across institutions or use of external/cloud computation requires site-specific approval. Therefore, Colab may only be used after written UHS and relevant hospital approvals explicitly permit the proposed de-identified data to be processed there. De-identification alone is not permission to upload.

## Current milestone

The research workspace, baseline inventory, provisional annotation schema, and 400-image experiment plan are recorded. The dataset builder, mask rasterizer, trainer, evaluation scripts, Colab notebook, and custom ONNX exporter are **not yet implemented**. No training or accuracy claim has been made.

Next milestone: freeze the annotation SOP and architecture decision, then implement and test the dataset builder against a small, approved pilot set.
