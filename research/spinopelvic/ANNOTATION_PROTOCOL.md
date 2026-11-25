# Draft annotation protocol — Spine-Contour-compatible v0.2

**Draft only.** The radiology readers and supervisor must approve anatomical definitions before production annotation.

## 1. Image and patient unit

- One coded standing lateral lumbopelvic examination per patient wherever possible.
- Store SHA-256 of the exact original image file bytes, width, height, orientation and pixel spacing when available. This matches the parent app's bundle-io.js sha256File implementation; do not hash decoded pixels or a re-encoded export.
- Coordinates use original decoded-image pixels: origin top-left, x rightward, y downward. For single-frame grayscale DICOM, match backend/server.py decoding: apply RescaleSlope/RescaleIntercept and MONOCHROME1 inversion; the annotation hash remains over the original DICOM file bytes.
- Store anatomical anterior-side mapping explicitly; do not infer A/P from screen position alone.
- Direct identifiers and the re-identification key must remain outside the research annotation dataset.

## 2. Masks for the segmentation models

### Vertebral U-Net target

Six classes: background (0), L1 (1), L2 (2), L3 (3), L4 (4), L5 (5).

Annotate five separate masks:
- L1
- L2
- L3
- L4
- L5

Keep the class identities stable. Do not relabel a visible L3 as L2 because another body is missing.

### Femoral U-Net target

Annotate one binary `FEMORAL_REGION_UNION` mask containing the visible combined projected femoral-head region. This matches the source model's one-foreground-class output.

Separately annotate two expert femoral-head centres as reference geometry. If two centres cannot be independently supported, record that status rather than inventing points. The union mask can still be retained for segmentation training, while PI/PT reference eligibility is withheld.

## 3. S1 detector target

For the two-keypoint Keypoint R-CNN, record:
- S1 detection bounding box
- S1 superior-endplate anterior endpoint (SA)
- S1 superior-endplate posterior endpoint (SP)

The bounding box must enclose the S1 detection target consistently with the approved SOP. Keypoint order is anatomical, after checking image orientation.

## 4. HRNet target: 22 landmarks

The existing HRNet checkpoint has 22 heatmaps:
- L1: SA, SP, IA, IP
- L2: SA, SP, IA, IP
- L3: SA, SP, IA, IP
- L4: SA, SP, IA, IP
- L5: SA, SP, IA, IP
- S1: SA, SP

The two femoral-head centres are additionally annotated for geometry reference and femoral fitter evaluation. They are not new HRNet channels in the Spine-Contour-aligned protocol.

## 5. Source annotation states

Use these states for every mask/keypoint object:
- `segmented`: model prediction reviewed and accepted unchanged; positive label.
- `edited`: model prediction corrected; corrected annotation is positive and original model prediction is retained as an error pair.
- `manual`: annotator places a structure missed by the model; positive label and explicit model miss.
- `not_placed`: structure is visible but has not been annotated; ignore, not negative.
- `not_visible`: annotator explicitly checked and confirmed the structure is absent/out of view; negative label where appropriate.

Never treat `not_placed` as `not_visible`. Uncertainty should be separately flagged. The state/provenance logic follows the Spine-Contour level-editor documentation, while the polygon masks are stored in the research-specific schema.

## 6. Reader reference values

For all test images, two trained clinical readers independently annotate the six essential spinopelvic measurement points and measure PI, PT, SS and LL. Store reader 1, reader 2 and adjudicated reference records separately. Do not use model predictions to create reference labels.

For development images, annotate the mask/keypoint targets required for model training and record the two femoral centres where independently supportable. Use the 30–50 image pilot to freeze definitions and repeatability/adjudication rules.

## 7. QC

- Check mask overlap, disconnected components, wrong vertebral class identity, polygon self-intersection, and out-of-image coordinates.
- Check L1/S1 endpoint order and image orientation.
- Check S1 box and keypoint geometry.
- Check the femoral union mask and the two-disc fitter's fit confidence/IoU and anatomical plausibility.
- Preserve original model output, corrected annotation, annotator, protocol version, image hash and timestamp.
- If femoral centre fitting fails QC, withhold PI/PT but retain valid SS/LL where possible.

## 8. Existing editor limitation

The Spine-Contour label record stores model/annotator state, geometry, original model geometry and selected mask contours. It does not provide a general hand-painted pixel-mask format for all corrected/manual masks. The UHS project therefore uses `schemas/annotation.schema.json` for the hand-drawn polygons and points, while preserving the source five-state annotation semantics.
