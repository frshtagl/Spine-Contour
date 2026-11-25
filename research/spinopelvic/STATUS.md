# Research status log

## 2026-11-25 — Workspace initialized

- Created isolated branch: `research/spinopelvic-lab`.
- Pinned baseline to fork commit `5df84e0af578e8faabab30b24e655735c3663ad4`.
- Recorded the current source architectures and pretrained Git LFS object hashes.
- Read the tentative UHS synopsis and mapped its two-model design to the source repository.
- Adopted the investigator's working cohort: 400 examinations, with 240 train / 60 validation / 100 locked test.
- Drafted a research annotation schema for separate masks, six required landmarks, and visibility/uncertainty states.
- Added safeguards to keep all patient data and training artifacts out of the public GitHub repository.

## Not done yet

- No patient images or masks received.
- No dataset has been built.
- No training code or Colab training notebook has been implemented.
- No checkpoint has been fine-tuned.
- No test-set evaluation has been performed.
- No custom ONNX model has been exported or integrated into the desktop application.

## Next actions

1. Approve/finalize the annotation SOP with the clinical readers.
2. Confirm institutional approval for de-identification and cloud/Colab processing.
3. Resolve whether Model A is nnU-Net v2, a compatible warm-start ResNet-34 U-Net, or a prespecified comparison of both.
4. Validate the source-image-to-model coordinate transform on approved pilot cases.
5. Implement dataset conversion and leakage checks.
6. Implement training, validation, evaluation, and export only after the data contract is stable.
