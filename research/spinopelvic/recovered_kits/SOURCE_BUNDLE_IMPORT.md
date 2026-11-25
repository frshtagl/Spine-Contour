# Source bundle import status

The source-only package is available in the user's ChatGPT Library and local storage:

- Filename: recovered_finetune_kits_source_only.tar.gz
- Size: 26,894 bytes
- SHA-256: aba797e420bbfc6bb98def2519d473690eadbacca3f2ade18d88ab37e0061629
- Contents: 27 source/configuration/README/provenance files
- Excludes: weights, sample radiograph, masks, annotations, overlays, per-case predictions and outputs

It preserves the vertebra U-Net, HRNet landmark and S1 Keypoint R-CNN fine-tuning source.

The branch now includes tools/import_source_bundle.sh. From the local repository root, download the archive from the Library and run:

~~~bash
bash research/spinopelvic/tools/import_source_bundle.sh /path/to/recovered_finetune_kits_source_only.tar.gz
~~~

The script verifies the expected SHA-256, rejects unsafe archive paths, refuses to overwrite an existing extraction and scans for weights/images/DICOMs or other case-level artifacts. It extracts into research/spinopelvic/recovered_kits/source_bundle/contents.

The archive is not expanded into individual tracked source files by the GitHub connector in this pass; it remains available in the Library/local PC. This is not a repository-visibility restriction. The archive is source-only and contains no weights or patient/sample data.

The source archive passed Python syntax compilation and JSON parsing. This is not a PyTorch checkpoint load, training run or clinical validation. No checkpoint weights are being re-uploaded.
