# Femoral kit — pending recovery

A femoral U-Net checkpoint pointer already exists in the fork:

- Path: `backend/weights/femoral_unet.pt`
- Git LFS pointer size: 97,878,776 bytes
- Pointer SHA-256: `4697c31fb39a7c31ac58e7a75c19e4cefebedb8e0a6fdb82e042e90190108779`

The user has not yet supplied a separate femoral fine-tuning ZIP in the recovered archive set inspected on 25 November 2026. The existing application also contains femoral inference implementation under `backend/femoral.py`.

When the femoral ZIP is found:
1. Inventory its code, requirements, metadata and checkpoint.
2. Compare its checkpoint SHA-256 and size with the upstream LFS pointer.
3. Preserve the fine-tuning source and metadata separately.
4. Exclude radiographs, masks, annotations, overlays and case-level predictions.
5. Do not overwrite the upstream baseline checkpoint.
