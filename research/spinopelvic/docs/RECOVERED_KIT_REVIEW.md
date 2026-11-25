# Recovered fine-tuning kits — technical review

## What is preserved

The user supplied three fine-tuning archives. A source-only package was prepared and validated as a 27-file archive:

- vertebra_unet_finetune/ — ResNet-34 U-Net, one grayscale channel, six classes (background + L1–L5).
- hrnet_landmarks_finetune/ — HRNet-W32, 22 heatmaps (four corners for each L1–L5 plus S1 superior-endplate SA/SP).
- s1_finetune/ — TorchVision Keypoint R-CNN, ResNet-50 FPN, two S1 keypoints.
- provenance/ — archive/checkpoint manifest and a hash-verifying checkpoint-only restore utility.

The source-only package contains no checkpoint binaries, sample radiographs, masks, annotations, overlays, per-case predictions or training outputs. The original ZIPs remain the local source of the recovered checkpoint files. **No weights are re-uploaded or duplicated in this work.**

## Checkpoint identity findings

| Model | Recovered checkpoint SHA-256 | Existing fork LFS pointer | Decision |
|---|---|---|---|
| Vertebra U-Net | b8dcb306e43fb1d13059c07cb4ec89d92d684d3d00444abac8543b9582acf5dd | Exact match | Reuse the existing fork checkpoint; no duplicate |
| HRNet | aa0f97a7f8dfe478b443ee6e87590185d5cd9278342538e8d4c67f7bdc243df9 | Different hash and size | Preserve as a distinct local checkpoint; do not overwrite baseline |
| S1 Keypoint R-CNN | 04c96dbd53a882d7cd387c0499f0526b23d364da7e7b8ca29ccf0378292130d2 | Different hash and size | Preserve as a distinct local checkpoint; do not overwrite baseline |
| Femoral U-Net | No separate recovered fine-tuning kit supplied yet | Existing fork pointer exists | New training entry point is provided; initialize from existing local fork checkpoint |

These are SHA-256 and byte-size comparisons against archive contents and LFS pointer metadata. They are not a claim that every LFS object was downloaded and strict-loaded in this environment.

## Source-kit review findings

The recovered training scripts are useful as model-specific reference implementations, but they should not be run directly against the whole cohort without the research split and dataset builder.

1. Their fallback split logic infers patient groups from filename prefixes such as patient__series__image. That is unsafe if filenames do not encode patient identity exactly. The research workflow now generates an explicit patient-disjoint split first and passes train/validation inputs separately.
2. The S1 kit derives its training bounding box from the two keypoints plus a configurable padding (BOX_PAD_PX, default 70 pixels). Its current KpDataset does not consume the manually annotated bbox field. The dataset builder retains the expert box for audit but records this training behavior explicitly.
3. The HRNet kit reports localization errors in original-image pixels and 768-frame pixels. The stored checkpoint's historical val_px unit is not independently established; use the new test report and explicit units.
4. The U-Net kit's historical val_dice definition is not guaranteed to match the source checkpoint's reported metric. Recompute the baseline on the current development validation set before comparing a fine-tune.
5. The source code and metadata are preserved as recovered research material. Syntax checks do not prove training correctness, checkpoint compatibility, or clinical validity.

## New research additions

- tools/make_annotation_templates.py: creates schema-valid empty annotation records from a coded CSV and local images.
- tools/prepare_splits.py: generates exact-count patient-disjoint train/validation/locked-test partitions; fails if the requested counts require splitting a patient.
- tools/build_training_inputs.py: verifies image-file SHA-256, rasterizes model targets, and writes the three recovered-kit dataset formats. Incomplete L1–L5 masks are excluded from U-Net training instead of being mislabeled as background. not_placed is never treated as a negative.
- tools/train_femoral_unet.py: adds a binary femoral-union fine-tuning loop using the source model's robust rescale, CLAHE, letterbox and 640-pixel model input. This is a newly specified study recipe, not the original hidden training recipe.
- tools/normalize_model_predictions.py: converts source HRNet/S1 prediction JSON and original-frame U-Net/femoral masks into separate model_point/model_bbox/model_polygons fields; human annotations remain separate.
- tools/compute_measurements_from_annotations.py: delegates PI/PT/SS/LL geometry to the exact fork function backend.utils.spinopelvic_measurements_from_geometry, with optional model-score filtering.
- analysis/analyze_agreement.py: reports coverage, MAE, median absolute error, bias, RMSE, Bland–Altman limits, ICC(A,1), and patient-cluster bootstrap intervals.
- tools/export_custom_checkpoint.py: strict-loads a selected custom checkpoint, exports ONNX, checks the graph and compares synthetic-input PyTorch/ONNX outputs.
- tools/validate_end_to_end_framing.py: synthetic round-trip checks for source S1 crop, point/mask restoration and reframe acceptance.

## Validation boundary

The new data utilities and statistical analysis were tested with synthetic images/records only. No real radiographs were opened, no recovered checkpoint tensors were loaded, no training job was run, and no clinical performance result was generated. The actual 240/60/100 split, source checkpoint compatibility, model fine-tuning and locked-test evaluation remain to be run locally after protocol approval and with approved data access.
