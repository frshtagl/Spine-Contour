# Training and evaluation plan — Spine-Contour-compatible

## Phase 0 — approvals and protocol freeze

1. Obtain UHS and relevant hospital ethics/archive approvals.
2. Obtain written site-specific approval for any inter-site data movement or Colab/cloud processing.
3. Approve the annotation SOP, model targets, image eligibility and adjudication rules.
4. Confirm the 400-case plan and patient-level split before test-set access.
5. Retrieve actual Git LFS checkpoint files and inspect their metadata/tensor shapes locally.
6. Pin source commit, checkpoint hashes, packages, random seeds and preprocessing.

## Phase 1 — cohort and split

| Split | n | Role |
|---|---:|---|
| Train | 240 | Gradient updates |
| Validation | 60 | Early stopping, hyperparameters and QC thresholds |
| Locked internal test | 100 | One-time final internal evaluation |
| Total | 400 | 300 development + 100 test |

The 300 development cases are annotated and split into train/validation. The 100 test cases receive independent expert reference labels but remain unavailable to training/model-selection code. Keep all examinations from a patient in one split. Prefer one examination per patient.

## Phase 2 — dataset builder

Implemented in tools/make_annotation_templates.py, tools/validate_annotations.py, tools/prepare_splits.py and tools/build_training_inputs.py. The local workflow:
- Validates records against `schemas/annotation.schema.json`.
- Joins image/annotation records by image SHA-256.
- Checks duplicate images and patient-group leakage.
- Rasterizes five mutually exclusive L1–L5 masks and the femoral union mask.
- Creates HRNet heatmap targets for 22 landmarks.
- Creates Keypoint R-CNN targets: S1 box plus two S1 keypoints.
- Writes both expert femoral centres to a separate geometry/reference_geometry_<split>.json file; they are not HRNet channels.
- Preserves original-image coordinates and delegates resize/letterbox preprocessing to the model-specific source trainers. Synthetic crop/restore and full locate/search tests are present in tools/validate_end_to_end_framing.py and tools/validate_synthetic_s1_search.py. Integration with the actual trained S1 detector remains pending.
- Spatial augmentation remains in the model-specific training code; verify that every image/mask/box/keypoint transform stays aligned.
- Records the split-manifest hash and per-file output hashes/fingerprint in the secure dataset-build manifest; test labels must not enter training loaders.

## Phase 3 — Model A: U-Net segmentation route

The recovered user-supplied vertebral U-Net training script is in the source-only archive; the new femoral U-Net trainer is in tools/train_femoral_unet.py. No training run has been performed.

- Fine-tune `backend/weights/vertebra_unet.pt` with L1–L5/background masks, after verifying the six-class channel semantics against the actual checkpoint.
- Fine-tune `backend/weights/femoral_unet.pt` with the binary femoral union mask.
- Fine-tune `backend/weights/s1_keypointrcnn.pt` with S1 bounding boxes and two S1 keypoints.
- Extract L1–L5 corners from the vertebral masks using the source corner extraction method.
- Apply the source two-disc fitter to the femoral union mask and validate centre errors against expert references.
- Calculate PI, PT, SS and LL with the shared geometry module.

## Phase 4 — Model B: HRNet landmark route

The recovered HRNet training script is in the source-only archive. No training run has been performed.

- Fine-tune the existing `backend/weights/hrnet_landmarks.pt` HRNet-W32 checkpoint with the existing 22 output channels.
- Preserve the original 22 channels: four corners for L1–L5 and two S1 endpoints.
- Do not add femoral-centre channels; use the shared femoral U-Net and two-disc fitter.
- Retain U-Net presence evidence as in the source pipeline, so HRNet cannot create unsupported vertebral levels without presence/QC evidence.
- Use the same S1 detector, femoral branch, image transforms, angle calculator and QC as Model A.

## Phase 5 — transfer-learning safeguards

The public source repository does not include the original lumbar training dataset or full training loop/hyperparameter recipe. The local fine-tuning procedure must therefore be specified and recorded as this study's protocol, not described as exact reproduction of the original training experiment.

- Start every experiment from the original source checkpoint; do not chain fine-tuned checkpoints.
- Evaluate unchanged source checkpoints as baselines.
- Use conservative learning-rate/encoder-freezing or regularization experiments on development data only.
- Use accepted `segmented`, corrected `edited`, and missed-but-manually-placed `manual` examples as positive labels.
- Treat `not_placed` as ignored; use `not_visible` as a negative only when explicitly confirmed.
- Maintain a frozen development benchmark and check for regression/catastrophic forgetting.
- Select optimizer, learning rate, batch size, epochs, loss weights and augmentation from development-only pilot experiments; freeze them before test evaluation.

## Phase 6 — locked evaluation

Report:
- Parameter-specific MAE, median absolute error, signed bias and patient-level bootstrap 95% CIs for PI, PT, SS and LL.
- Per-parameter coverage, four-angle coverage and withholding reasons.
- Bland–Altman bias and limits of agreement; ICC(A,1) with CIs.
- Paired Model A/Model B absolute-error differences on the same test patients with valid outputs; report coverage differences separately.
- L1–L5 class-wise Dice, femoral union-mask Dice, line-angle error, keypoint localization error, femoral-centre error and runtime.
- Screened, excluded, reference-ineligible, attempted, accepted and withheld denominators.

Missing/withheld values must never be set to zero. A model must not appear more accurate merely because it abstains on difficult cases.

## Phase 7 — ONNX export and versioning

The custom-checkpoint exporter is implemented at tools/export_custom_checkpoint.py. It strict-loads the state dictionary, checks the ONNX graph and compares PyTorch/ONNX outputs on synthetic inputs. This is not real-image parity or clinical validation.

- Run the exporter on each selected custom checkpoint.
- Export each selected checkpoint and run ONNX graph validation.
- Compare PyTorch and ONNX outputs numerically on synthetic/approved development examples.
- Record source checkpoint SHA-256, ONNX SHA-256, config, dataset-manifest hash, annotation version and software versions.
- Register new model IDs; do not overwrite the original source model.
- Keep baseline and fine-tuned models selectable with a rollback path.

## Colab

Colab is conditional on written UHS and site-specific permission. If approval is absent, use institution-controlled compute. Never commit patient images, masks, annotations, reference readings or weights to this public repository.


## Implementation status — 25 November 2026

The deterministic split generator, schema-checked dataset builder, source-compatible measurement extraction, agreement analysis, femoral U-Net training entry point and custom-checkpoint ONNX exporter are now present in the research branch. The U-Net/HRNet/S1 training loops are preserved in the source-only archive and imported locally using the hash-verifying script. Actual strict checkpoint loading, fine-tuning, end-to-end predicted S1 framing, Colab orchestration and locked-test evaluation remain pending.
