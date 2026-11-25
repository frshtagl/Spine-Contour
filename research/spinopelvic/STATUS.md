# Research status log

## 2026-11-25 — Spine-Contour alignment revision

- Reviewed the full UHS spinopelvic synopsis.
- Audited the public Spine-Contour source architecture and annotation/retraining documentation.
- Corrected the earlier scaffold: Model A is now the source ResNet-34 U-Net route, not nnU-Net v2.
- Model B uses the existing 22-channel HRNet-W32; the previously proposed 24-channel extension is removed.
- Added the shared S1 Keypoint R-CNN (ResNet-50 FPN, two keypoints).
- Changed the femoral target to one combined region mask plus source-style two-disc fitting; two expert centres remain as reference geometry.
- Recommended 400 examinations: 300 development (240 train + 60 validation) and 100 locked test.
- Updated annotation protocol, schema, example and tests to include L1–L5 masks, femoral union mask, S1 box/keypoints and all 22 HRNet landmarks.
- Prepared synopsis amendment recommendations and ready-to-insert wording.

## Important source limitation

The public repository includes the lumbar model architectures/checkpoints and inference/export implementation, but not the original lumbar training data or complete lumbar training loop/optimization recipe. Our local fine-tuning recipe must be documented as this study's method, not described as exact replication of the original hidden training experiment.

## Not done (initial scaffold baseline; superseded by the later continuation entry)

- No study patient images or annotations were uploaded to this chat/repository at the initial scaffold stage.
- No actual Git LFS checkpoint tensors inspected in this environment.
- At that initial stage, no dataset builder/training loop/Colab notebook was implemented.
- No fine-tuning or test evaluation performed.
- No performance or clinical validation claim made.

## Next actions (initial scaffold baseline; superseded by the later continuation entry)

1. Approve the synopsis changes and annotation SOP with supervisor/readers/statistician.
2. Confirm 240/60/100 or revise the allocation.
3. Confirm UHS/site approval for the planned compute route.
4. Download LFS checkpoints locally and verify checkpoint metadata/tensor shapes.
5. Validate schema and coordinate transforms on synthetic cases and an approved pilot.
6. Implement dataset builder, trainers, evaluation and ONNX export.


## 2026-11-25 — repository cross-check

- Verified personal fork frshtagl/Spine-Contour and branch research/spinopelvic-lab.
- Branch head at audit: dd9e2c2; 10 commits ahead of main; 17 tracked research files. The research branch is not empty; it is separate from main.
- Added the exact source-derived application annotation strategy and a dated repository gap audit.
- Added tools/audit_research_branch.py to check research-file presence, missing pipeline components and Git LFS pointer status without opening patient data.
- Confirmed the branch is still a scaffold: patient split generation, dataset construction, model training loops, evaluation, custom ONNX export and executable Colab workflow were absent at the audited head.
- No patient data were accessed; no checkpoint tensor was strict-loaded; no model was trained; no clinical result is claimed.

## 2026-11-25 — recovered kit audit and repository staging

- Reopened the three original user-supplied archives from the Library and verified their exact byte sizes and SHA-256 fingerprints.
- Inspected ZIP contents without extracting case-level image/annotation payloads into Git. The archives contain a sample lateral radiograph and/or masks, pseudo-annotations, overlays, predictions or measurements; those are excluded from the public repository.
- Reopened the source-only package: 27 source/configuration/documentation files, no checkpoint binaries and no case-level image/label assets.
- Ran Python syntax compilation on all 14 Python files in the source-only package: passed. Parsed all JSON metadata: passed. This is not a PyTorch checkpoint load or training test.
- Compared recovered checkpoint hashes against the fork's current Git LFS pointer files. U-Net matches exactly; HRNet and S1 differ in both size and hash. Preserve them as separate recovered checkpoints; do not overwrite upstream baseline paths.
- Confirmed an existing femoral U-Net LFS pointer at `backend/weights/femoral_unet.pt`; the separate femoral ZIP is still pending.
- Added explicit Git LFS tracking patterns and a narrow allow-list for recovered checkpoint staging paths. No recovered checkpoint binaries were uploaded.
- Added checkpoint upload instructions, per-model weight staging READMEs, a provenance manifest and a femoral-kit placeholder.
- Repository visibility was rechecked: still public. No patient-level data, sample images, masks, annotations or recovered checkpoint binaries were committed.
- The source-only archive remains in the user's ChatGPT Library until the fork is private and kit-specific provenance/licensing is reviewed.
- No checkpoint tensor loading, model training, test evaluation or clinical validation was performed.


## 2026-11-25 — research pipeline continuation (supersedes earlier scaffold status)

- Added a coded annotation-template generator and strengthened validation to check polygon area, self-intersection, model geometry and image-frame bounds.
- Corrected an important source-derived detail: the parent app hashes the exact original file bytes, not decoded greyscale pixels. The protocol, exact-strategy document and annotation schema now say so.
- Added single-frame grayscale DICOM support matching backend/server.py: RescaleSlope/RescaleIntercept and MONOCHROME1 inversion; DICOM PixelSpacing is mapped into x/y order. Model inputs are written as float TIFF while the manifest hash remains tied to the original DICOM bytes.
- Added deterministic patient-disjoint split generation with exact 240/60/100 counts, patient-group leakage checks, exact subset-sum validation allocation and frozen manifest hashes.
- Added a schema-checked dataset builder for vertebral U-Net, femoral U-Net, 22-channel HRNet and S1 Keypoint R-CNN. It excludes incomplete U-Net masks, preserves not_placed versus not_visible, verifies source image hashes/dimensions and physically separates optional locked-test inputs.
- Added source-compatible measurement extraction through backend.utils.spinopelvic_measurements_from_geometry.
- Added agreement reporting for coverage, MAE, median absolute error, bias, RMSE, Bland–Altman limits, ICC(A,1) and patient-cluster bootstrap confidence intervals.
- Added a femoral U-Net training entry point aligned to the parent preprocessing path and a custom-checkpoint ONNX exporter with strict state loading, graph checking and synthetic-input parity checks.
- Added reference-measurement and split-manifest schemas, experiment configuration/log templates, model cards, protocol decision register and a local source-bundle importer.
- Added synthetic regression tests and a GitHub Actions workflow for this research branch.
- Updated the recovered-kit policy to respect the user's instruction: no checkpoint weights are re-uploaded or duplicated. Existing baseline Git LFS paths remain untouched; recovered HRNet/S1 variants remain local.
- Repository remains public. Only code, schemas, docs and synthetic tests are committed. No study cohort data, DICOMs, masks, annotations, patient-level manifests, fine-tuned checkpoints or clinical results were added.
- GitHub Actions synthetic research suite: 19 tests passed on commit 0d850002954aa9427de4d03e04eee197bc7c85a9 ([run](https://github.com/frshtagl/Spine-Contour/actions/runs/45559482964)). The Pillow deprecation warnings were removed in subsequent commits; rerun CI after those final edits.
- No real study radiographs were accessed in this continuation, no LFS tensor checkpoint was strict-loaded, no training run was performed and no clinical validation is claimed.

## Remaining work

1. Import the source-only recovered-kit archive locally with tools/import_source_bundle.sh (no weights are imported).
2. Run the branch CI and the synthetic tests on the user's local clone.
3. Strict-load the actual local baseline and recovered HRNet/S1 checkpoints; compare model state keys/shapes and resolve any incompatibility before training.
4. Run tools/validate_end_to_end_framing.py and tools/validate_synthetic_s1_search.py locally with the parent backend dependencies; then add an actual trained-S1-detector-to-crop integration test and exercise the normalizer against the recovered source-kit prediction files.
5. Freeze the approved protocol, reader/adjudication plan and seed; generate the real patient-disjoint split in secure storage.
6. Run a small approved pilot, then development-only training/selection. Open the locked test only once after all thresholds, QC and analysis are frozen.
7. Colab orchestration remains optional and must not be used unless the institution/site explicitly approves the proposed de-identified data transfer and processing.
