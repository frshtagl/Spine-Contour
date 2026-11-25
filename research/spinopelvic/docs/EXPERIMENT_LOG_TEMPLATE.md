# Experiment record — [experiment ID]

> Store this completed record in approved secure research storage. Do not commit patient-level paths, identifiers, logs or checkpoint binaries to the public repository.

## Identity and reproducibility

- Protocol version:
- Git commit:
- Start/end UTC:
- Operator code:
- Random seed:
- Python / PyTorch / TorchVision / CUDA versions:
- Environment lockfile SHA-256:
- Split manifest SHA-256:
- Annotation schema/protocol version:
- Training reader/adjudication code:

## Data

- Training / validation counts:
- Patient-group disjointness verified:
- Exclusions and reasons:
- Image hash algorithm: SHA-256 of exact source file bytes
- Locked test opened: NO / YES (if YES, attach authorization and frozen analysis-plan version)

## Model

- Model slot:
- Architecture / encoder / head:
- Initialization checkpoint path label:
- Initialization checkpoint SHA-256:
- Strict state-dict load: PASS / FAIL
- Preprocessing revision:
- Input dimensions and coordinate frame:
- Augmentation:
- Optimizer / learning rate / scheduler / weight decay:
- Epochs / batch size / early stopping:

## Validation results

- Baseline result on the same validation set:
- Best epoch / selection criterion:
- Segmentation or landmark metrics:
- Measurement MAE / bias / RMSE / ICC / coverage:
- Withheld outputs and reasons:
- Subgroup checks (site, image orientation, image quality, relevant clinical subgroups):
- Failure review and reviewer sign-off:

## Artifacts

- Best checkpoint SHA-256:
- ONNX SHA-256:
- Training log SHA-256:
- Export graph validation:
- Real-image parity status:
- Notes / deviations / protocol amendments:

## Decision

- Continue / reject / revise:
- Rationale:
- Approved by:
- Date:
