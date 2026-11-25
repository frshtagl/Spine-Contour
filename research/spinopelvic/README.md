# Spine-Contour Spinopelvic Research Workspace

**Status:** research scaffold and synopsis amendment prepared; no patient data or model training runs have been added.

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

The public repository includes architecture builders and ONNX export code, but not the original lumbar training dataset or a complete lumbar training loop/recipe. We will therefore fine-tune compatible checkpoints using a newly documented local protocol, not claim exact reproduction of undisclosed original training hyperparameters.

Each experiment starts from the original checkpoint, not a chain of previously fine-tuned checkpoints. Keep the unchanged source checkpoint as a baseline and maintain a frozen development benchmark to detect regression/catastrophic forgetting.

## Data security

The parent repository is public. Never commit radiographs, DICOMs, masks, annotation JSON, patient manifests, reader measurements, trained checkpoints or derived patient-level artifacts. The nested `.gitignore` blocks common data/artifact paths. Keep data and weights in institution-approved secure storage.

The synopsis requires site-specific approval for cross-institutional transfer and external/cloud computation. Colab may be used only after written UHS and relevant hospital approvals explicitly permit the proposed processing route.

## Current milestone

- Research workspace and initial annotation tools are present.
- Spine-Contour-compatible architecture and synopsis amendments are documented.
- The revised annotation schema now supports the source architecture's mask and landmark targets.
- Dataset builder, training loops, evaluation scripts, Colab notebook and custom ONNX exporter remain to be implemented.
- No training, test evaluation or accuracy claim has been made.

Next: approve the synopsis/SOP, retrieve and inspect the actual LFS checkpoints, validate the annotations on a small approved pilot, and implement the deterministic dataset builder.


## Repository branch and audit note

The spinopelvic workspace is on the separate branch research/spinopelvic-lab, not on main. Open the [research folder on that branch](https://github.com/frshtagl/Spine-Contour/tree/research/spinopelvic-lab/research/spinopelvic).

The exact source-derived editor method is recorded in docs/ANNOTATION_STRATEGY_EXACT.md. The repository state and missing implementation components are recorded in docs/REPOSITORY_CROSS_CHECK_2026-11-25.md.

Run python research/spinopelvic/tools/audit_research_branch.py from a clone to check expected files and Git LFS checkpoint status. The research branch remains a scaffold until the missing split, dataset, training, evaluation, export and end-to-end framing components are implemented and tested.
