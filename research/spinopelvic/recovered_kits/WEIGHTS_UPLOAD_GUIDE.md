# Checkpoint policy — no duplicate weight upload

This fork is public and GitHub reports that visibility cannot be changed for this fork. The current research task therefore keeps the repository code-only.

## Explicit decision

**Do not re-upload any trained weights.** The original application checkpoints remain in their existing Git LFS paths:

- backend/weights/vertebra_unet.pt
- backend/weights/hrnet_landmarks.pt
- backend/weights/s1_keypointrcnn.pt
- backend/weights/femoral_unet.pt

The recovered U-Net checkpoint is byte-identical to the existing U-Net LFS pointer. Re-uploading it would create a duplicate. The recovered HRNet and S1 files have different hashes from the existing LFS pointer versions; preserve those recovered files on the user's local PC or approved private model storage and do not overwrite the upstream baselines.

## Local use

From a trusted local clone, inspect current pointers and fetch only the baseline files needed for a run:

~~~bash
git lfs install
git lfs ls-files
git lfs pull --include="backend/weights/vertebra_unet.pt,backend/weights/hrnet_landmarks.pt,backend/weights/s1_keypointrcnn.pt,backend/weights/femoral_unet.pt"
~~~

Verify the local file hashes against the approved manifest before loading. Strict state-dictionary loading must pass before any fine-tuning begins. The connected GitHub session did not download or tensor-load the LFS binaries during this repository organization work.

## New experiment outputs

Fine-tuned checkpoints, ONNX exports, training logs, per-study predictions and split manifests belong in institution-approved secure local storage. This public research branch contains code, schemas, synthetic fixtures, tests and aggregate documentation only.

Do not commit radiographs, DICOMs, masks, annotation JSON, patient-group manifests, reader measurements or per-study results, even when the images have been de-identified.

## Existing baseline protection

- Never overwrite backend/weights/*.pt with a recovered checkpoint.
- Start each experiment from a recorded baseline checkpoint hash.
- Do not chain one fine-tuned model into another unless it is a separate pre-registered experiment.
- Preserve original checkpoint hashes and run manifests.
- No weight upload is required to continue the current work.
