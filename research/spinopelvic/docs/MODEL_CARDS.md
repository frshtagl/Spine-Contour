# Research model cards — spinopelvic pipeline

These are research component descriptions, not claims of clinical validation.

## Vertebral segmentation — U-Net

- Architecture: segmentation_models_pytorch U-Net, ResNet-34 encoder.
- Input: raw S1-framed crop from the parent app's reframe/fallback logic, then one grayscale channel letterboxed to 768 × 768.
- Output: six-class label map (background, L1, L2, L3, L4, L5).
- Target: separate polygon masks for L1–L5.
- Initialization: existing backend/weights/vertebra_unet.pt; recovered user checkpoint hash matches the existing LFS pointer exactly.
- Key risks: missing/partial labels, adjacent-level overlap, wrong level ordering, cropped levels and annotation inconsistency.
- Required reporting: per-level Dice, mean foreground Dice, boundary review, missed-level rate and source-coordinate landmark error.

## Vertebral landmarks — HRNet-W32

- Architecture: existing HRNet-W32.
- Input: raw S1-framed crop from the parent app's reframe/fallback logic, then one grayscale channel letterboxed to 768 × 768.
- Output: 22 heatmaps: four corners for each L1–L5 and two S1 superior-endplate endpoints.
- Target: point annotations; missing/not-placed points are masked from the heatmap loss by the recovered dataset code.
- Initialization: existing backend/weights/hrnet_landmarks.pt; recovered user checkpoint is a distinct file and must not overwrite the existing baseline.
- Key risks: orientation, low contrast, cropped endpoints, coordinate-space confusion and heatmap peak ambiguity.
- Required reporting: per-landmark pixel error in original-image coordinates, P95 error, miss rate, orientation-stratified performance and end-to-end crop performance.

## S1 detector — Keypoint R-CNN

- Architecture: TorchVision Keypoint R-CNN, ResNet-50 FPN, two S1 superior-endplate keypoints.
- Input: original unframed radiograph/candidate search windows, three repeated grayscale channels letterboxed to 768 × 768. The S1 detector remains unframed in the training-input builder because it localizes the candidate window.
- Target: S1 box and SA/SP endpoints.
- Initialization: existing backend/weights/s1_keypointrcnn.pt; recovered user checkpoint is a distinct file and must not overwrite the existing baseline.
- Key risks: missed S1, endpoint swaps, box truncation, overlapping structures and crop-localization failure.
- Required reporting: detection recall, keypoint error, S1 endplate angle error, endpoint swap rate and full-frame end-to-end failures.

## Femoral region — binary U-Net plus two-disc fitter

- Architecture: one-channel ResNet-34 U-Net with one foreground channel.
- Input: raw S1-framed crop, robust rescale, CLAHE (2.0, 8×8), 768 letterbox and 640 × 640 model input; validation flip averaging; threshold 0.35.
- Target: one combined femoral-head union mask, not two separate instance masks.
- Initialization: existing backend/weights/femoral_unet.pt.
- Centre extraction: source backend.femoral.fit_two_discs union-of-two-discs fit.
- New training/inference entry points: tools/train_femoral_unet.py and tools/predict_femoral_unet.py. This is a newly specified study recipe, not an exact reproduction of an unpublished historical training run.
- Key risks: near-total head overlap, edge truncation, pelvis/femur superposition, poor union-mask fit and false hip-axis certainty.
- Required reporting: positive-reference union-mask Dice/IoU, empty-reference specificity, centre error, fit acceptance, edge-truncation rate and PI/PT withholding rate. Empty-reference cases are reported separately and are not counted as Dice=1 to inflate the positive-case score.

## Shared geometry and QC

The source-frame cropper is tools/make_source_frames.py and calls the fork's backend.framing.reframe/fallback_window. HRNet points and U-Net/femoral masks are restored to original radiograph coordinates using frame_manifest_all.json before geometry. All spinopelvic angles should be computed with the fork's existing backend.utils.spinopelvic_measurements_from_geometry implementation. PI/PT require S1 and two femoral centres; SS requires S1; LL requires L1 and S1. Withhold only the unsupported parameter(s), preserve independent valid results and report coverage.
