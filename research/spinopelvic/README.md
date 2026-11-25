# Spine-Contour Spinopelvic Research Workspace

**Status:** reproducible research utilities and synthetic CI tests added; no study-cohort data, training runs or clinical results are stored in this repository.

This isolated workspace adapts the available Spine-Contour lumbar/lumbopelvic architecture and workflow for the UHS spinopelvic research study. The application on `main` remains unchanged.

## Study endpoints

- Pelvic incidence (PI)
- Pelvic tilt (PT)
- Sacral slope (SS)
- L1–S1 lumbar lordosis (LL)

## Working cohort plan

| Partition | Images | Role |
|---|---:|---|
| Training | 240 | Gradient updates only |
| Validation | 60 | Model/threshold selection and early stopping |
| Locked internal test | 100 | Final independent internal evaluation |
| **Total** | **400** | **300 development + 100 locked test** |

The 300 development examinations are expert-annotated and split into 240 training and 60 validation cases. The 100 test examinations also require independent expert reference annotations, but those labels are kept separate and are not used for training, model selection, threshold calibration or iterative debugging. Splits are patient-disjoint; one examination per patient is preferred.

This is the recommended interpretation of the investigator's “300 annotated + 100 validation/test” plan. It keeps a 100-case final test set while creating a validation set within the 300 development cases. The synopsis's illustrative precision calculation should be updated and approved by the statistician.

## Spine-Contour-compatible model architecture

The proposed study follows the architecture visible in the public Spine-Contour source:

### Model A — U-Net vertebral segmentation route

- `segmentation_models_pytorch` U-Net with ResNet-34 encoder
- Single grayscale input
- Six output classes: background plus L1–L5
- Extract vertebral corners/endplate lines from the predicted masks

### Model B — HRNet vertebral landmark route

- Existing HRNet-W32 checkpoint
- 22 heatmaps: four corners for each of L1–L5 and two S1 endplate points
- Use the existing 22-channel head; do not add two femoral-centre channels
- Retain U-Net presence evidence as in the reference inference path

### Shared components

- S1 localization/endplate: TorchVision Keypoint R-CNN, ResNet-50 FPN, two keypoints
- Femoral region: ResNet-34 U-Net, one foreground output channel for the combined femoral-head region
- Femoral centre extraction: two-disc fitting on the combined mask, using distance-transform/ridge/peeling seeds, robust circle fitting and union-mask IoU optimization
- Shared angle calculator, QC rules, source-coordinate restoration and locked test cases

The two vertebral routes are alternatives within the same overall pipeline. They are not two entirely independent systems: both use the same S1 and femoral components and the same geometry/QC modules.

## Preprocessing to reproduce

- Robust grayscale rescaling using the source implementation's 0.5th and 99.5th percentiles
- Aspect-ratio-preserving letterbox to 768 × 768 for the shared vertebral/S1/HRNet model frame
- S1-relative crop localization and source-coordinate restoration
- Separate femoral input path: CLAHE before padding/resizing, 640 × 640 input, and horizontal-flip probability averaging
- Verify crop/letterbox forward and inverse coordinate transforms before training

## Annotation design

The research annotation format is deliberately separate from the current Spine-Contour level-editor record because the editor record is not a general hand-painted pixel-mask format.

Required hand annotations:
- Separate L1, L2, L3, L4 and L5 masks for the six-class vertebral U-Net
- One combined femoral-head union mask for the one-class femoral U-Net
- S1 detection box and two S1 superior-endplate keypoints for Keypoint R-CNN
- 22 landmark points for the HRNet target: four corners for each L1–L5 plus two S1 points
- Two independent femoral-head centres as reference geometry for validating the two-disc fitter and calculating PI/PT

Follow Spine-Contour's annotation-state semantics:
- `segmented`: model output accepted
- `edited`: model output corrected; preserve the original prediction as an error pair
- `manual`: anatomy placed by hand after a model miss
- `not_placed`: visible but not annotated; ignore for training
- `not_visible`: explicitly checked and absent; negative label where appropriate

A fused femoral mask is the expected segmentation target. The two-disc fitter must pass prespecified quality checks against expert reference centres; otherwise PI/PT are withheld.

## Training and provenance

Available checkpoints are in the repository's Git LFS paths:
- `backend/weights/vertebra_unet.pt`
- `backend/weights/hrnet_landmarks.pt`
- `backend/weights/s1_keypointrcnn.pt`
- `backend/weights/femoral_unet.pt`

The parent source includes architecture builders and inference/export code, but not the original lumbar training dataset or complete original training recipe. Separate user-supplied recovered fine-tuning kits provide U-Net, HRNet and S1 training scripts; their source-only archive remains in the user's Library/local storage and can be imported with tools/import_source_bundle.sh. This branch adds the missing cohort split, dataset preparation, femoral fine-tuning, geometry/agreement analysis and custom ONNX export utilities. These are research implementations, not claims to reproduce undisclosed original hyperparameters. No trained weights are duplicated or re-uploaded.

Each experiment starts from the original checkpoint, not a chain of previously fine-tuned checkpoints. Keep the unchanged source checkpoint as a baseline and maintain a frozen development benchmark to detect regression/catastrophic forgetting.

## Data security

The parent repository is public. Never commit radiographs, DICOMs, masks, annotation JSON, patient manifests, reader measurements, trained checkpoints or derived patient-level artifacts. The nested `.gitignore` blocks common data/artifact paths. Keep data and weights in institution-approved secure storage.

The synopsis requires site-specific approval for cross-institutional transfer and external/cloud computation. Colab may be used only after written UHS and relevant hospital approvals explicitly permit the proposed processing route.

## Current milestone

- Architecture alignment, synopsis recommendations and the research annotation schema are documented.
- Annotation template generation, schema/polygon validation, patient-disjoint split generation and model-specific dataset construction are implemented.
- The dataset builder supports raster images and single-frame grayscale DICOM using the parent application's slope/intercept and MONOCHROME1 behavior.
- Source-compatible geometry extraction, agreement statistics, a new femoral U-Net fine-tuning entry point and custom-checkpoint ONNX export are implemented.
- Synthetic tests and branch-specific CI are added.
- Recovered U-Net/HRNet/S1 trainers remain in the source-only archive in the user's Library/local storage; import them locally with the committed hash-verifying script.
- No real study data were accessed, no checkpoint tensor was strict-loaded, no training was run and no clinical accuracy claim has been made.

Next: import the source-only kit locally, run CI, strict-load the existing local checkpoints, obtain protocol/reader approval, pilot the annotations and then freeze the real patient-disjoint split.

## Repository branch and audit note

The spinopelvic workspace is on the separate branch research/spinopelvic-lab, not on main. Open the [research folder on that branch](https://github.com/frshtagl/Spine-Contour/tree/research/spinopelvic-lab/research/spinopelvic).

The exact source-derived editor method is recorded in docs/ANNOTATION_STRATEGY_EXACT.md. The repository state and missing implementation components are recorded in docs/REPOSITORY_CROSS_CHECK_2026-11-25.md.

Run python research/spinopelvic/tools/audit_research_branch.py from a clone to check expected files and Git LFS pointer status. The core data-preparation and agreement utilities are now present; Colab orchestration, end-to-end framing/coordinate-transform tests, strict local checkpoint loading and real-data validation remain.

## Recovered source kits and checkpoint staging

See [recovered kits](recovered_kits/README.md) for the three user-supplied fine-tuning packages, archive/checkpoint fingerprints and upstream LFS comparison. No checkpoint binaries are re-uploaded. This public research branch contains code, schemas, synthetic tests and documentation only; keep all case-level data and newly fine-tuned model artifacts in approved secure storage.


## Added pipeline tools (25 November 2026)

- tools/make_annotation_templates.py — creates schema-validated empty annotation records from coded CSV metadata.
- tools/validate_annotations.py — checks schema, bounds, polygon area/self-intersection and model/corrected geometry.
- tools/prepare_splits.py — exact-count patient-disjoint 240/60/100 split; fails on patient overlap or impossible group allocation.
- tools/build_training_inputs.py — validates source hashes/dimensions, rasterizes masks, adapts the 22-channel HRNet and S1 training JSONs, and physically separates locked-test inputs.
- tools/make_source_frames.py — uses the parent app's exact S1 reframe/fallback functions to create crop-frame U-Net/HRNet/femoral inputs; S1 detector data remain unframed.
- tools/normalize_model_predictions.py — converts recovered HRNet/S1 predictions and U-Net/femoral crop masks back into original-image model_point/model_bbox/model_polygons while keeping human-reviewed fields separate.
- tools/compute_measurements_from_annotations.py — calls the fork's existing spinopelvic geometry function, including model-point records.
- analysis/analyze_agreement.py — coverage, MAE, bias, RMSE, Bland–Altman, ICC(A,1) and patient-cluster bootstrap CIs.
- tools/train_femoral_unet.py — new femoral-union fine-tuning recipe aligned to the source preprocessing path.
- tools/export_custom_checkpoint.py — strict-load, ONNX graph check and synthetic-input PyTorch/ONNX parity.
- tools/validate_end_to_end_framing.py — synthetic S1 crop, inverse point/mask and reframe acceptance checks.
- tools/validate_synthetic_s1_search.py — exercises the full locate/search/fixed-point selection path with a deterministic synthetic scorer; not a trained detector test.
- .github/workflows/spinopelvic-research-tests.yml — synthetic-only regression tests.

The research utilities write to secure local output paths outside the Git repository. The 100-case locked test is excluded by default and must only be opened after protocol and model-selection freeze. Model outputs are normalized into a separate prediction annotation record; raw model coordinates/scores never overwrite human-reviewed annotations.
