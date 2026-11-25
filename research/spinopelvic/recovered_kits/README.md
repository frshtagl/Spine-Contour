# Recovered fine-tuning kits and checkpoint provenance

This folder is the isolated preservation area for the three user-supplied fine-tuning kits. It is separate from the application implementation under `backend/`.

## Current repository state

- Repository: [frshtagl/Spine-Contour](https://github.com/frshtagl/Spine-Contour)
- Working branch: `research/spinopelvic-lab`
- Visibility at last check: **PUBLIC**
- The upstream application already has model checkpoints tracked through Git LFS under `backend/weights/`.
- Do not upload recovered weights or any patient-linked data until the repository owner changes visibility to private.

## Preserved kits

| Kit | Architecture | User-supplied checkpoint |
|---|---|---|
| Vertebra U-Net | ResNet-34 U-Net, six classes (background + L1-L5) | `vertebra_unet.pt` |
| HRNet landmarks | HRNet-W32, 22 heatmaps | `landmarks_hrnet.pt` |
| S1 keypoints | Keypoint R-CNN, ResNet-50 FPN, two keypoints | `s1_keypointrcnn.pt` |

The source-only package is currently retained in the user's ChatGPT Library as `recovered_finetune_kits_source_only.tar.gz`. It contains 27 source/configuration/documentation files, no checkpoint binaries, and no sample images or labels. See `SOURCE_BUNDLE_IMPORT.md`.

## Important checkpoint comparison

The supplied U-Net checkpoint exactly matches the existing `backend/weights/vertebra_unet.pt` Git LFS pointer by SHA-256 and size.

The supplied HRNet and S1 checkpoints **do not** match the current fork's LFS pointer objects. Keep them as separate recovered-kit checkpoints. Do not overwrite the upstream baseline paths.

The fork also has an existing femoral U-Net LFS pointer at `backend/weights/femoral_unet.pt`; the separate femoral fine-tuning ZIP has not yet been received.

See `ARTIFACT_MANIFEST.json` for sizes, hashes and exact paths.

## Planned structure

```text
recovered_kits/
├── vertebra_unet_finetune/
│   └── weights/                 # recovered U-Net checkpoint after private switch
├── hrnet_landmarks_finetune/
│   └── weights/                 # recovered HRNet checkpoint after private switch
├── s1_finetune/
│   └── weights/                 # recovered S1 checkpoint after private switch
├── femoral/
│   └── weights/                 # pending femoral kit
├── source_bundle/               # source-only archive import
├── provenance/
└── ARTIFACT_MANIFEST.json
```

The code/configuration archive will be imported after privacy is changed and its kit-specific provenance/licensing is reviewed. The three weight directories are prepared with LFS rules but no binaries have been added.

## Data protection

The original ZIPs contain a sample lateral radiograph and case-linked masks, annotations, overlays or measurements. Those files are deliberately excluded. A private repository is still not a substitute for institution-approved clinical-data storage. Keep patient images, labels, manifests, and reader measurements in approved secure storage; do not place them in Git or Git LFS.

## Validation status

The source-only package passed Python syntax compilation and JSON parsing. This is not a full execution test: PyTorch checkpoint loading, training, validation and clinical evaluation have not been performed. The recovered scripts are preservation material, not a validated clinical pipeline.

## License

The parent Spine-Contour project declares AGPL-3.0. The recovered kits do not include a separate license or complete provenance statement. Confirm terms before redistributing weights or derivative models.
