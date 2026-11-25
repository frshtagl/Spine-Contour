# Training and evaluation plan

## Phase 0 — freeze the data contract

Before training:
1. Approve the annotation SOP and schema with the clinical readers.
2. Confirm institutional permission for the image source, de-identification, transfer, and Colab use.
3. Confirm the exact pixel preprocessing and coordinate transforms.
4. Pin package versions, source commit, model checkpoint hashes, and random seeds.
5. Build patient-level train/validation/test manifests and lock the test partition.

## Phase 1 — dataset builder

Implement a deterministic dataset builder that:

- Validates annotation JSON against `schemas/annotation.schema.json`.
- Resolves each coded case to the exact approved local image using SHA-256.
- Checks duplicate images and patient-group leakage.
- Validates polygon geometry and landmark bounds.
- Rasterizes masks without changing aspect ratio or orientation.
- Applies the same spatial transform to image, masks, polygons, and landmarks.
- Creates train/validation data only from the 300 development cases.
- Keeps the 100 test cases and references inaccessible to the training/tuning process.
- Writes a manifest hash and a clear report of excluded/invalid cases.

## Phase 2 — Model A: segmentation-to-geometry

The synopsis names 2D nnU-Net v2 as the primary segmentation route. The repository's existing segmentation checkpoints are segmentation_models_pytorch ResNet-34 U-Nets, not nnU-Net v2. Do not load their state dictionaries into nnU-Net or claim that this is direct fine-tuning.

Run a documented architecture feasibility gate:

- **Warm-start experiment:** fine-tune the exact existing Spine-Contour ResNet-34 U-Net(s) using only target channels proven compatible with their original checkpoint semantics. The existing femoral model has one foreground channel and can use a union mask. The existing vertebral model has six output channels; verify the precise class/channel mapping before training.
- **Synopsis-aligned experiment:** build the specified 2D nnU-Net v2 segmentation pipeline. Treat the existing ResNet-34 U-Net weights as a separate baseline unless a tensor-compatible transfer path is demonstrated and tested.
- Select and document the primary Model A implementation before opening the locked test set. If the final method differs from the synopsis, amend the synopsis with supervisor/statistician approval.

Geometry extraction must use explicit L1 and S1 superior-endplate lines and two independently supported femoral-head centres. A combined femoral mask centroid is prohibited as a substitute for the bicoxofemoral midpoint. Record per-angle pass/withheld status and the exact QC reason.

## Phase 3 — Model B: direct landmarks

Use the existing HRNet-W32 checkpoint as the initialization candidate.

The existing checkpoint has 22 heatmap channels. A candidate transfer plan is:
- Preserve the original 22 channels and their learned weights.
- Add two heatmap channels for FH1_CENTER and FH2_CENTER.
- Initialize the new channels explicitly.
- Fine-tune on the six mandatory points plus the original vertebral landmarks where labels are available.
- Keep the original 22-channel model untouched as the baseline.

The model must expose independent femoral centres and uncertainty/withholding logic. A heatmap peak alone is not proof that the head is independently visible.

## Phase 4 — training controls

- Use only the 240 training cases for gradient updates.
- Use only the 60 validation cases for early stopping, architecture selection, hyperparameters, and QC-threshold calibration.
- Never use the 100 locked test cases for model selection or iterative debugging.
- Start from a clean copy of the source checkpoint for each experiment; do not chain fine-tunes without recording ancestry.
- Record optimizer, learning rate, schedule, batch size, epochs, augmentations, loss weights, random seed, software versions, GPU, and checkpoint-selection rule.
- Use conservative augmentation that preserves anatomical plausibility and transforms all annotations consistently.
- Compare against the unchanged source model on the same locked cases.
- Preserve a best-validation checkpoint and a final reproducible training log.

Exact hyperparameters are intentionally not fixed in this scaffold; choose them through development-only pilot experiments and record them before test evaluation.

## Phase 5 — locked evaluation

For each model, report:
- Parameter-specific MAE, median absolute error, signed bias, and 95% confidence intervals for PI, PT, SS, and LL.
- Per-parameter valid-output coverage and withholding reasons.
- Complete four-angle coverage among reference-eligible test cases.
- Bland–Altman bias and limits of agreement.
- ICC(A,1) with confidence intervals.
- Paired absolute-error differences on the same patients where both models return valid results; report coverage differences separately.
- Landmark error and segmentation Dice where applicable.
- L1/S1 line-angle error, femoral-centre error, runtime, and failure categories.

Report all denominators. Do not set withheld outputs to zero. Do not compare MAE without also reporting coverage; a model must not look better merely by abstaining on difficult cases.

## Phase 6 — export and application integration

After a model passes the predeclared internal benchmark:
1. Export a custom ONNX graph from the selected PyTorch checkpoint.
2. Add a custom-checkpoint option to the exporter.
3. Check ONNX graph validity and numerical parity against PyTorch.
4. Verify source-image coordinate restoration on real approved radiographs.
5. Add a new model ID; do not overwrite the original model.
6. Record checkpoint SHA-256, ONNX SHA-256, training-data manifest hash, annotation version, and evaluation report.
7. Keep original and custom models selectable, with rollback.

No model should be described as clinically validated or safe for diagnostic use on the basis of this internal retrospective experiment.
