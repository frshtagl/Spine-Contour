# Synopsis alignment — Spine-Contour-compatible protocol

## Existing synopsis reviewed

File: `UHS_Spinopelvic_Synopsis_Revised.docx` (20 pages; surfaced 25 November 2026).

The current synopsis already defines PI, PT, SS and L1–S1 LL; six essential geometric measurement points; a 30–50 image reader pilot; independent readers; adjudication; parameter-specific withholding; MAE, coverage, Bland–Altman and ICC reporting. The principal amendments are to the model architecture, annotation targets, cohort allocation and transfer-learning description.

## Recommended cohort interpretation

| Partition | n | Purpose |
|---|---:|---|
| Training | 240 | Model parameter updates |
| Validation | 60 | Checkpoint/threshold/hyperparameter selection |
| Locked internal test | 100 | Final independent evaluation |
| Total | 400 | 300 development + 100 test |

All 300 development cases are annotated and split into training/validation. The 100 test cases receive independent expert reference annotations, but are not exposed to training or model-selection code. Patient-level separation is mandatory. If the investigator instead means 300 training-only plus 100 combined validation/test, the validation/test split must be specified and test precision will be lower if only 50 are reserved for test.

The existing synopsis's illustrative calculation gives n ≈ 115 for SD = 3° and a 95% CI half-width of 0.55°. Under the same illustration, n = 100 gives a half-width of approximately 0.59°; n = 50 gives approximately 0.83°. A statistician must approve the final precision target.

## Replace the nnU-Net v2 primary architecture

The public Spine-Contour lumbar path uses `segmentation_models_pytorch` U-Net with ResNet-34 encoder, one grayscale input and six output classes interpreted as background plus L1–L5. It does not use nnU-Net v2 for this lumbar path. If the investigator wants the same architecture, replace the synopsis's nnU-Net v2 wording with this ResNet-34 U-Net.

## Model A — segmentation route

- Vertebral segmentation: ResNet-34 U-Net, six classes (background + L1–L5).
- S1: two-keypoint ResNet-50 FPN Keypoint R-CNN for S1 localization/endplate points.
- Femoral region: one-class ResNet-34 U-Net for the combined femoral-head union mask.
- Femoral centres: source-style two-disc fitting on the union mask, with independent expert centre references and prespecified fit QC.
- Geometry: shared fixed angle calculator for PI, PT, SS and LL.

## Model B — landmark route

- Existing HRNet-W32, 22 heatmaps: four corners for each of L1–L5 plus two S1 points.
- Keep the existing 22-channel checkpoint; do not add two femoral-centre channels.
- Retain U-Net presence evidence as in the source inference path.
- Use the same S1 and femoral branches, crop/preprocessing, geometry calculator and QC as Model A. This isolates the vertebral geometry route.

## Femoral target correction

The source femoral model predicts one combined region mask. The source postprocessor fits two discs to the merged component. Therefore the primary annotation target should be `FEMORAL_REGION_UNION`, not two independent semantic head masks. Annotators should additionally place the two independent expert reference centres (and optionally radii/rim points) so that centre extraction can be evaluated. A failed or implausible two-disc fit must cause PI/PT withholding; it must not be treated as a valid hip axis.

## Annotation format

Use separate hand-drawn L1, L2, L3, L4 and L5 masks; one femoral union mask; an S1 detection box and two S1 keypoints; all 22 HRNet landmark points; and two expert femoral-centre points. Preserve the five source annotation states: segmented, edited, manual, not_placed and not_visible. The current source level-editor record is geometry/provenance oriented, not a complete manual pixel-mask editor; a separate research mask schema is required.

## Transfer-learning statement

State that compatible available Spine-Contour checkpoints will be fine-tuned on the local development cohort. Do not claim exact replication of the original lumbar training procedure: the public repository lacks the original lumbar training data, complete training loop and full optimization recipe. Start each experiment from the original checkpoint, keep the unchanged source checkpoint as a baseline, and freeze the test set.

## Proposed synopsis edits

1. Change tentative sample size from 420 (300 development + 120 test) to 400 (300 development + 100 test), with 240 training and 60 validation within the 300.
2. Replace nnU-Net v2 with ResNet-34 U-Net for Model A if exact architecture alignment is the priority.
3. Replace the previous 24-channel HRNet proposal with the existing 22-channel HRNet-W32.
4. Add the shared S1 Keypoint R-CNN and femoral U-Net/two-disc fitting components.
5. Replace two separate femoral segmentation masks with one union mask plus two expert centre points.
6. Add source-style annotation-state semantics and preserve model-original/corrected annotations.
7. Add explicit transfer-learning and model-version/ONNX provenance details.
8. Preserve the synopsis's ethics requirement: Colab/cloud use needs written UHS and site-specific approval.
9. Update the sample-size precision statement and obtain statistician approval before locking the 100 test cases.
