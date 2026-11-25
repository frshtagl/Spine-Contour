# Spine-Contour baseline inventory — spinopelvic adaptation

## Pinned source

- Working repository: `frshtagl/Spine-Contour`
- Upstream project: `Feches/Spine-Contour`
- GitHub-reported code license: AGPL-3.0
- Research branch base commit: `5df84e0af578e8faabab30b24e655735c3663ad4`
- Source paths are reused; pretrained weights are not duplicated in this research folder.

## Existing lumbar/lumbopelvic checkpoints

The Git LFS pointer metadata exposes the following object hashes and sizes. These are pointer values; the actual tensors must be downloaded with Git LFS and inspected before training.

| File | Source role | LFS SHA-256 | Bytes |
|---|---|---|---:|
| `backend/weights/vertebra_unet.pt` | L1–L5 segmentation | `b8dcb306e43fb1d13059c07cb4ec89d92d684d3d00444abac8543b9582acf5dd` | 97,881,592 |
| `backend/weights/femoral_unet.pt` | Combined femoral-head region | `4697c31fb39a7c31ac58e7a75c19e4cefebedb8e0a6fdb82e042e90190108779` | 97,878,776 |
| `backend/weights/hrnet_landmarks.pt` | Vertebral landmark heatmaps | `2cc7f791c9a0df3b84515ddb065e52ccff954b8264b4374b4cbdfe8f71870e36` | 127,888,626 |
| `backend/weights/s1_keypointrcnn.pt` | S1 two-keypoint detector | `2d54fe9f26f2d9bd1f1968b5bfa43c1b6fce5d87aaff8ed495ec5312c7cfc685` | 236,592,704 |

## Exact source architectures

### Vertebral segmentation U-Net

- Builder: `backend/models/training.py::build_unet`
- Library: `segmentation_models_pytorch`
- Architecture: U-Net, ResNet-34 encoder
- Input: one grayscale channel
- Output: six classes, interpreted by the lumbar path as background plus L1–L5
- Shared model frame: 768 × 768
- Inference: class argmax, presence extraction, quadrilateral corner extraction and mapping back to source pixels

### HRNet vertebral landmarks

- Builder: `backend/models/hrnet.py::HRNetLandmarks`
- Backbone: HRNet-W32 through `timm`
- Input: one grayscale channel
- Output: 22 heatmaps — four corners (SA, SP, IA, IP) for each of L1–L5 plus two S1 superior-endplate points
- Heatmap stride: 4
- Decoder: sub-pixel soft-argmax around the heatmap peak
- In the app, HRNet is an alternative source for vertebral corners; U-Net evidence is retained as a presence/validity check.

Do not extend this checkpoint to 24 outputs for the spinopelvic project. Femoral centres are supplied by the separate femoral branch and its geometric fitter.

### S1 detector

- Builder: `backend/models/training.py::build_s1_model`
- TorchVision Keypoint R-CNN, ResNet-50 FPN
- Two keypoints
- Used to locate/reframe the lumbosacral region and return the S1 superior-endplate endpoints
- Shared model frame: 768 × 768

### Femoral segmentation and fitting

- Builder: same ResNet-34 U-Net builder, one foreground output channel
- Target: a combined/union mask of the projected femoral-head region, not two separate semantic instance channels
- Femoral input path: CLAHE on native pixels, then letterbox and resize to 640 × 640
- Inference uses horizontal-flip probability averaging and a 0.35 mask threshold in the source implementation
- `backend/femoral.py::fit_two_discs` fits two circles to the union mask using distance-transform peaks, ridge ends and peel-based seeds; robust circle refinement and union-mask IoU optimization select a fit. The result includes fit confidence and IoU.

The study must independently validate the two fitted centres against expert reference points. Do not accept two circles solely because the optimizer returned them.

## Preprocessing and framing

- `backend/models/models.py::_robust_rescale`: percentile-based grayscale rescaling (0.5th/99.5th percentiles).
- `backend/models/models.py::_letterbox`: preserve aspect ratio on a square canvas with explicit scale/padding transform.
- `backend/framing.py`: S1-relative lumbar crop, candidate search, fallback and source-coordinate restoration.
- Shared model canvas: 768 × 768.
- Femoral model canvas: 640 × 640 after its CLAHE path.
- All geometric augmentations must transform image pixels, masks, bounding boxes and keypoints consistently.
- Implement forward/inverse coordinate parity tests before any training.

## Annotation workflow facts

The source level editor records five states:
- `segmented`: model output accepted
- `edited`: model output corrected; original geometry retained
- `manual`: structure placed after a model miss
- `not_placed`: visible but not annotated; ignored
- `not_visible`: explicitly checked absent; negative label

Its record contains per-film SHA-256, image dimensions, model/version, geometry, selected contours and edit summaries. It does not provide a general arbitrary hand-painted mask representation for edited/manual vertebrae. The research project therefore uses its own polygon/raster mask schema while retaining the same state semantics and model-original provenance.

## Training/export facts and limits

- `backend/models/training.py` builds architectures and loads checkpoints for export/validation; it is not a complete training loop.
- The public repository does not include the original lumbar training images, a lumbar dataset manifest, or a complete lumbar training recipe/hyperparameter record.
- `docs/superpowers/NEXT-SESSION-retraining.md` is a design note for a future retraining workflow, not evidence that that workflow has already been implemented.
- The public cervical provenance file is for a separate cervical model/dataset and must not be used as lumbar training provenance.
- `tools/export_onnx.py` exports the original checkpoints and performs graph/parity checks. A custom-checkpoint exporter or CLI option must be added for fine-tuned weights.

## Reproducibility

Record source commit, LFS checkpoint SHA-256, annotation protocol version, image hash, patient-level split manifest hash, preprocessing version, YAML config, software versions, GPU, random seeds, optimizer/schedule, training log, selected checkpoint hash, ONNX hash and evaluation report hash.
