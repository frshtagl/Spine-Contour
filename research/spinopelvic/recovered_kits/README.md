# Recovered fine-tuning kits

This folder preserves an inventory and cryptographic fingerprints for three user-supplied kits: vertebral U-Net, HRNet landmarks and S1 Keypoint R-CNN.

## Public-repository safety

This repository is public. The original ZIPs contained a lateral radiograph, pseudo-masks, annotations, overlays and per-case predictions/measurements. Those case-level materials were deliberately excluded. The three large PyTorch checkpoints were also not uploaded; two exceed GitHub's ordinary 100 MB file limit. Keep original archives and weights in approved private storage and verify them against ARTIFACT_MANIFEST.json.

## Source-only code bundle

A sanitized source-only archive was prepared in the conversation as recovered_finetune_kits_source_only.tar.gz (26,894 bytes; SHA-256 aba797e420bbfc6bb98def2519d473690eadbacca3f2ade18d88ab37e0061629). It contains Python scripts, READMEs, requirements and sanitized model metadata, plus a script that extracts only the three checkpoints from the original ZIPs and verifies their hashes. It contains no weights or case-level data. The connected GitHub integration cannot push binary artifacts or Git LFS objects; the archive therefore still needs a manual upload if it should live in this repository.

## Models

- Vertebral U-Net: ResNet-34, six classes (background + L1-L5).
- HRNet landmarks: HRNet-W32, 22 heatmaps.
- S1 Keypoint R-CNN: ResNet-50 FPN, two S1 keypoints.

See ARTIFACT_MANIFEST.json for exact archive/checkpoint sizes and SHA-256 hashes. This is preservation metadata, not a claim of clinical validation. The parent project declares AGPL-3.0; the recovered kits' independent licensing/provenance has not been established.
