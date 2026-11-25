# Spine-Contour baseline inventory

## Pinned source

- Working repository: `frshtagl/Spine-Contour`
- Upstream project: `Feches/Spine-Contour`
- Repository license reported by GitHub: AGPL-3.0
- Baseline branch: `main`
- Baseline commit: `5df84e0af578e8faabab30b24e655735c3663ad4`
- This research branch starts from that commit. The original application files are not modified by this scaffold.

## Existing lumbar/lumbopelvic model assets

The following model files are stored at the repository root under Git LFS. The values below are the LFS object SHA-256 and byte size reported by the Git LFS pointer, not a claim that the binary weights were loaded or independently inspected during this audit.

| File | Existing role | LFS object SHA-256 | Bytes |
|---|---|---|---:|
| `backend/weights/vertebra_unet.pt` | Vertebral segmentation | `b8dcb306e43fb1d13059c07cb4ec89d92d684d3d00444abac8543b9582acf5dd` | 97,881,592 |
| `backend/weights/femoral_unet.pt` | Femoral-region segmentation | `4697c31fb39a7c31ac58e7a75c19e4cefebedb8e0a6fdb82e042e90190108779` | 97,878,776 |
| `backend/weights/hrnet_landmarks.pt` | Lumbar/S1 landmark heatmaps | `2cc7f791c9a0df3b84515ddb065e52ccff954b8264b4374b4cbdfe8f71870e36` | 127,888,626 |
| `backend/weights/s1_keypointrcnn.pt` | S1 keypoint detector | `2d54fe9f26f2d9bd1f1968b5bfa43c1b6fce5d87aaff8ed495ec5312c7cfc685` | 236,592,704 |

Do not duplicate these large weight files inside this research directory. The pinned root paths are the source of truth. Retrieve actual LFS objects in a local clone with Git LFS before attempting training. The connector inspection exposed LFS pointer metadata, not the actual checkpoint tensors.

## Architectures established from source code

### Vertebral segmentation

- Builder: `backend/models/training.py::build_unet`
- Library: `segmentation_models_pytorch`
- Architecture: U-Net with a configurable encoder; checkpoint defaults to ResNet-34
- Input: one grayscale channel
- Output configured by the loader: six classes
- Inference canvas: 768 × 768 in the shared model configuration

The exact six-channel target ordering must be confirmed from the complete inference-to-label mapping and checkpoint metadata before constructing a training target. Do not assume channel order from the output count alone.

### Femoral-region segmentation

- Same U-Net builder, configured for one foreground class
- Femoral model canvas constant: 640 × 640
- The downstream code in `backend/femoral.py` explicitly describes two overlapping femoral heads being fitted from one segmented blob.

This is useful for a warm-start experiment: the existing one-class mask can be trained on the union of the visible femoral-head regions, while separate expert contours and centre labels are retained in the research annotations. If two centres cannot be independently supported, PI and PT must be withheld.

### Direct landmark model

- Builder: `backend/models/hrnet.py`
- Backbone: HRNet-W32 via `timm`
- Input channels: one
- Existing output: 22 heatmaps — four corners for each of L1–L5 plus two S1 superior-endplate points
- Heatmap stride: stored in the checkpoint; verify when loading the actual LFS checkpoint
- Decoder: sub-pixel heatmap decoding is already implemented

A 24-channel extension is a feasible transfer-learning experiment, not an existing feature. Copy the trained weights for the 22 existing output channels, initialize two new femoral-centre channels, and test strict loading and decoding. Preserve the original 22-channel model unchanged as the baseline.

### S1 detector

- Builder: TorchVision Keypoint R-CNN with ResNet-50 FPN
- Two keypoints
- Existing checkpoint is available through Git LFS.
- It is not automatically the same as the six-point direct-landmark comparator.

## Existing preprocessing to reproduce

- `backend/models/models.py` contains robust grayscale rescaling and 768 × 768 letterboxing.
- `backend/framing.py` contains lumbar/femoral crop selection and coordinate-restoration transforms.
- Femoral inference uses its own model-size and crop path.
- Training augmentation must transform pixels, masks, polygons, and landmarks identically.
- Before training, write a parity test that maps a known point from source-image coordinates into model coordinates and back, with no material coordinate drift.

Do not simply resize all source films to a square. Reproduce the relevant inference framing or document and validate a deliberate change.

## Existing annotation support and its limitation

The level editor supports L1–L5 corners, S1 endplate points, and two femoral heads represented by centre/radius and optional rim points. Its record distinguishes:

- `segmented`: unchanged model geometry
- `edited`: model geometry corrected by the annotator
- `manual`: anatomy placed by hand after a miss
- `not_visible`: explicitly marked absent
- `not_placed`: visible but not annotated; ignored for training

Important limitation: the existing record is not a general hand-painted pixel-mask format. Body contours are retained for unchanged segmented predictions; edited/manual entries primarily preserve geometric points. Femoral heads are represented by circles/rim points, not complete hand-drawn binary masks. Therefore, this project adds a separate polygon-mask annotation schema in `schemas/annotation.schema.json`.

## Training/export gaps

- `backend/models/training.py` builds architectures and loads existing checkpoints; it is not a complete training loop.
- There is no general mask dataset builder, split generator, trainer, experiment tracker, or full evaluation pipeline in the audited source paths.
- `tools/export_onnx.py` exports the repository's fixed checkpoint paths; a custom-checkpoint argument and custom-model metadata/parity checks will be needed.
- Model registration currently exposes the original model IDs. A research model must be registered under a new ID and remain optional until validated.

## Reproducibility rule

Every experiment must record: source commit, LFS checkpoint hashes, dataset-manifest hash, annotation protocol version, patient-level split manifest hash, preprocessing version, model config, random seeds, package versions, training log, best-checkpoint hash, and evaluation report hash.
