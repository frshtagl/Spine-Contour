# Source-only bundle staging

The archive recovered_finetune_kits_source_only.tar.gz is preserved in the user's ChatGPT Library and local storage.

- Size: 26,894 bytes
- SHA-256: aba797e420bbfc6bb98def2519d473690eadbacca3f2ade18d88ab37e0061629
- Contains 27 source, configuration, README and provenance files for vertebra U-Net, HRNet landmarks and S1 Keypoint R-CNN.
- Excludes weights, radiographs, masks, annotations, overlays and case-level outputs.

Use the branch's tools/import_source_bundle.sh to verify and extract it into this folder's contents subdirectory. This archive is source-only; it does not restore, duplicate or upload any model weights.

The repository is public. The code-only source archive is safe to stage after provenance/license review; all patient-linked data and model binaries remain outside Git.
