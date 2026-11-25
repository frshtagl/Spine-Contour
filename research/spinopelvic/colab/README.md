# Google Colab workflow — Spine-Contour-aligned training

This is a planned workflow, not an executable training notebook yet.

## Data approval

The UHS synopsis requires site-specific approval for cross-institutional movement and external/cloud computation. Do not upload radiographs, DICOMs, masks, annotation JSON, patient manifests or reference readings to Colab/Drive until written UHS and relevant hospital approvals explicitly authorize that processing route. Use institution-controlled compute if cloud processing is not approved.

## Intended sequence after approval

1. Clone the research branch.
2. Retrieve the original checkpoint files using Git LFS.
3. Install pinned research/export dependencies.
4. Read the approved, de-identified dataset from institution-approved storage outside the Git repository.
5. Validate the annotation JSON and image hashes.
6. Create patient-disjoint splits: 240 training, 60 validation, 100 locked test.
7. Build targets:
   - L1–L5/background masks for the vertebral ResNet-34 U-Net
   - femoral union mask for the femoral ResNet-34 U-Net
   - S1 bounding boxes and two keypoints for Keypoint R-CNN
   - 22 heatmaps for HRNet-W32
8. Fine-tune from the original checkpoints; retain the original source model as baseline.
9. Save checkpoints/logs to approved persistent storage, not the public repository.
10. Use validation only for selection; open the 100-case test set only after the pipeline and analysis plan are frozen.
11. Export selected weights to ONNX and run graph/numerical parity tests.
12. Commit code/configs/synthetic tests and approved aggregate results only.

## Training provenance

The public repository does not contain the original lumbar training set or complete training recipe. Record the local optimizer, learning rate, batch size, epochs, augmentation, loss, random seed, software/GPU versions and checkpoint-selection rule. Start each experiment from the original source checkpoint rather than chaining fine-tunes.

## Pending implementation

- Full dataset builder and patient-level split generator
- U-Net segmentation training loops
- HRNet heatmap training loop
- Keypoint R-CNN S1 training loop
- Custom checkpoint loading/export options
- Locked evaluation and statistical report scripts
- Executable Colab notebook with approval/data-location preflight checks
