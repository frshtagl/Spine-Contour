# Femoral kit — source checkpoint available; separate fine-tuning kit pending

A femoral U-Net checkpoint pointer already exists in the fork:

- Path: backend/weights/femoral_unet.pt
- Git LFS pointer size: 97,878,776 bytes
- Pointer SHA-256: 4697c31fb39a7c31ac58e7a75c19e4cefebedb8e0a6fdb82e042e90190108779

The user has not supplied a separate femoral fine-tuning ZIP in the recovered archive set inspected on 25 November 2026. The existing application contains femoral inference under backend/femoral.py and preprocessing in backend/models/models.py.

A new local training entry point is now present at research/spinopelvic/tools/train_femoral_unet.py. It uses the existing one-channel ResNet-34 U-Net architecture and mirrors the source inference preprocessing: robust rescale, CLAHE (2.0, 8x8), 768 letterbox, 640 resize, grayscale normalization, horizontal-flip validation TTA and the source threshold 0.35.

This is a newly specified fine-tuning recipe, not a recovered historical femoral training script. It strict-loads the existing local checkpoint, writes new run artifacts to a secure output directory outside Git, and refuses to overwrite a non-empty run folder.

When a separate femoral kit is found:
1. Inventory its code, requirements, metadata and checkpoint.
2. Compare its checkpoint SHA-256 and size with the upstream LFS pointer.
3. Preserve fine-tuning source and metadata separately.
4. Exclude radiographs, masks, annotations, overlays and case-level predictions.
5. Do not overwrite or re-upload the existing baseline checkpoint.
