# Synopsis amendment recommendations — Spine-Contour alignment

Prepared 25 November 2026 against `UHS_Spinopelvic_Synopsis_Revised.docx` and the public source at https://github.com/frshtagl/Spine-Contour.

See the companion Word amendment pack supplied in the conversation for full ready-to-insert paragraphs.

## Recommended study allocation

- Total: 400 examinations.
- Development: 300, expert annotated; 240 training + 60 validation.
- Locked internal test: 100 separate examinations with independent expert reference annotations.
- All splits patient-disjoint. Test labels remain inaccessible to training/model-selection code.

This is the recommended interpretation of “300 annotated + 100 validation/test”: 100 is the locked test set, while validation is held within the 300 development cases. If 300 is intended to mean training-only, a different allocation must be approved; 300 train + 50 validation + 50 test weakens final test precision.

Using the synopsis's illustrative SD of 3°, approximate 95% CI half-width is 0.59° for n=100, 0.55° for n=115, and 0.83° for n=50. The statistician must approve the final target.

## Match the available Spine-Contour architecture

1. **Model A — U-Net segmentation:** `segmentation_models_pytorch` U-Net, ResNet-34 encoder, one grayscale input, six classes (background + L1–L5).
2. **Model B — HRNet landmarks:** existing HRNet-W32 checkpoint, 22 heatmaps (four corners per L1–L5 plus two S1 points). Keep 22 outputs; do not add femoral-centre channels.
3. **Shared S1:** Keypoint R-CNN with ResNet-50 FPN and two keypoints for S1 localization/endplate geometry.
4. **Shared femoral:** one-class ResNet-34 U-Net for the combined femoral-head region mask; estimate two centres with the source two-disc fitting algorithm and QC against expert reference centres.
5. Both routes share preprocessing, source-coordinate restoration, geometry calculator, QC and test set.

## Replace in synopsis

- Replace nnU-Net v2 with the source ResNet-34 U-Net for Model A.
- Replace proposed 24-channel HRNet with the source 22-channel HRNet.
- Replace two separate femoral segmentation masks with one `FEMORAL_REGION_UNION` mask plus two independently annotated expert centres.
- Add S1 Keypoint R-CNN and femoral two-disc fitting.
- Specify 300 development (240 train/60 validation) + 100 locked test.
- Describe fine-tuning from available checkpoints, with original-model baseline, no chained fine-tunes, validation-only selection and ONNX parity/provenance.
- Keep the source five annotation states: segmented, edited, manual, not_placed (ignored), not_visible (explicit negative).
- Preserve the existing ethics condition: Colab/cloud requires written institutional/site approval.

## Critical qualification

The public repository contains model builders, checkpoints in Git LFS, inference/preprocessing and export code. It does not contain the original lumbar training data or full lumbar training loop/hyperparameter recipe. The synopsis may claim adaptation of the available Spine-Contour architecture and fine-tuning of its compatible checkpoints; it must not claim exact replication of undisclosed original training hyperparameters/data.
