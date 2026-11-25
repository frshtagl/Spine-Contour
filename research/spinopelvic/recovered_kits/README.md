# Recovered fine-tuning kits and checkpoint provenance

This is the isolated preservation area for the three user-supplied fine-tuning kits. It is separate from the application implementation under backend/.

## Repository and data policy

- Repository: [frshtagl/Spine-Contour](https://github.com/frshtagl/Spine-Contour)
- Working branch: research/spinopelvic-lab
- Repository visibility: PUBLIC (the owner reports GitHub does not allow changing visibility for this fork).
- By explicit instruction, **no recovered weights are re-uploaded or duplicated**.
- This branch receives code, schemas, synthetic tests, documentation and research utilities only. Do not commit case-level images, DICOMs, masks, annotations, split manifests, reader measurements, per-study predictions or trained artifacts.

## Preserved kits

| Kit | Architecture | User-supplied checkpoint |
|---|---|---|
| Vertebra U-Net | ResNet-34 U-Net, six classes (background + L1-L5) | vertebra_unet.pt |
| HRNet landmarks | HRNet-W32, 22 heatmaps | landmarks_hrnet.pt |
| S1 keypoints | Keypoint R-CNN, ResNet-50 FPN, two keypoints | s1_keypointrcnn.pt |

The source-only archive is available in the user's ChatGPT Library and local storage as recovered_finetune_kits_source_only.tar.gz. It contains 27 source/configuration/documentation files, no checkpoint binaries and no sample/case-level images or labels. The branch includes tools/import_source_bundle.sh to verify its SHA-256 and extract it locally into source_bundle/contents. It has not been expanded into individual tracked source files in this commit.

## Checkpoint comparison

The supplied U-Net checkpoint exactly matches the existing backend/weights/vertebra_unet.pt Git LFS pointer by SHA-256 and size. Do not duplicate it.

The supplied HRNet and S1 checkpoints differ from the current fork's LFS pointer objects. Preserve them as separate local recovered checkpoints; do not overwrite the upstream baseline or upload them again.

The fork already has a femoral U-Net LFS pointer at backend/weights/femoral_unet.pt. No separate femoral fine-tuning ZIP has been supplied. A new local training entry point is available at research/spinopelvic/tools/train_femoral_unet.py.

See ARTIFACT_MANIFEST.json for archive sizes, hashes, architectures and checkpoint comparison.

## Local source bundle import

Download the source-only archive from the ChatGPT Library to your local PC, then run from the repository root:

~~~bash
bash research/spinopelvic/tools/import_source_bundle.sh /path/to/recovered_finetune_kits_source_only.tar.gz
~~~

The importer verifies the expected SHA-256, checks archive paths, refuses to overwrite an existing extraction and rejects weights/images/DICOMs/clinical-data artifacts. It extracts source only; it does not restore or upload checkpoints.

## Validation boundary

The source-only archive passed Python syntax compilation and JSON parsing. This does not prove checkpoint loading, training correctness or clinical validity. No recovered checkpoint tensor was loaded and no training or clinical evaluation was performed in this environment.

## License and provenance

The parent Spine-Contour project declares AGPL-3.0. The recovered kits do not include a separate license or complete provenance statement. Review the upstream and kit-specific terms before redistributing source or derivative models. This does not require re-uploading the weights.
