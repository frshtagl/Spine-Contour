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

## Not done

- No patient images or annotations received.
- No actual Git LFS checkpoint tensors inspected in this environment.
- No dataset builder/training loop/Colab notebook implemented.
- No fine-tuning or test evaluation performed.
- No performance or clinical validation claim made.

## Next actions

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
