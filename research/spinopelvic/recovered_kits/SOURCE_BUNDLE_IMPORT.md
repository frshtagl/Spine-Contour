# Source bundle import status

The source-only package is available in the user's ChatGPT Library:

- Filename: `recovered_finetune_kits_source_only.tar.gz`
- Size: 26,894 bytes
- SHA-256: `aba797e420bbfc6bb98def2519d473690eadbacca3f2ade18d88ab37e0061629`
- Contents: 27 source/configuration/README/provenance files
- Excludes: model weights, sample radiograph, masks, annotations, overlays, per-case predictions and outputs

It preserves the three model kits:

1. `vertebra_unet_finetune/`
2. `hrnet_landmarks_finetune/`
3. `s1_finetune/`

The source archive has passed Python syntax compilation and JSON parsing. A scan found no sample case identifier or credential-like string.

**Why it is not yet copied into this repository:** the fork is still public, and the recovered kits do not include a separate license or complete provenance statement. The archive will be added to `source_bundle/` after the repository is private and the provenance/licensing decision is confirmed.

Extract locally with:

```bash
tar -xzf recovered_finetune_kits_source_only.tar.gz
```
