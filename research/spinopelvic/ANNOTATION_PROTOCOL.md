# Draft annotation protocol — version 0.1

**Draft only.** Clinical readers and the supervisor must approve the anatomical definitions before production annotation.

## 1. Unit of annotation

- One coded examination per patient wherever possible.
- Store source-image width and height and SHA-256 of the exact pixel data used.
- Coordinates use the original stored image frame: x increases rightward, y increases downward, origin at the top-left.
- Record image orientation/anterior side explicitly; do not infer anterior/posterior from screen position without the orientation field.
- No names, hospital numbers, dates of birth, accession numbers, or other direct identifiers in annotation files.

## 2. Required segmentation masks

Provisional mask targets:

| Target | Meaning |
|---|---|
| L1 | Visible L1 vertebral body region |
| L2 | Visible L2 vertebral body region |
| L3 | Visible L3 vertebral body region |
| L4 | Visible L4 vertebral body region |
| L5 | Visible L5 vertebral body region |
| S1_SUPERIOR_ENDPLATE | Visible superior S1 endplate region, not an invented full S1 body mask |
| FEMORAL_REGION_UNION | Union of visible femoral-head region(s), for a one-class femoral segmentation warm-start |
| FEMORAL_HEAD_1 | Separate head contour where individually traceable |
| FEMORAL_HEAD_2 | Separate head contour where individually traceable |

The first seven targets are the provisional mutually exclusive semantic segmentation labels for a segmentation model. The two individual head contours are retained as separate annotation objects for geometric fitting and quality control. They must not be fabricated when the heads are fused or indistinguishable.

This is a draft target list, not a claim that the existing six-output vertebral checkpoint already uses this exact mapping. The source channel order must be verified before any pretrained segmentation weights are loaded.

## 3. Six mandatory landmarks

Each study should receive these points when anatomically supportable:

- `L1_SUP_ANT`: anterior endpoint of the superior L1 endplate.
- `L1_SUP_POST`: posterior endpoint of the superior L1 endplate.
- `S1_SUP_ANT`: anterior endpoint of the superior S1 endplate.
- `S1_SUP_POST`: posterior endpoint of the superior S1 endplate.
- `FH1_CENTER`: centre of independently identified projected femoral head 1.
- `FH2_CENTER`: centre of independently identified projected femoral head 2.

Use the anatomical anterior/posterior labels after checking the stored image orientation. Head 1/head 2 should have a stable image-space ordering recorded in the SOP; do not label them left/right anatomically unless the projection and metadata justify that mapping.

## 4. Visibility and uncertainty

For every mask and landmark, distinguish:

- `annotated`: visible and confidently outlined/placed.
- `partial`: only a defensible portion is visible and is annotated as such.
- `not_visible`: the reader confirms the structure is outside the field of view or absent from the image.
- `not_separable`: overlapping anatomy prevents independent identification (especially femoral heads).
- `not_annotated`: no decision has been made; this is not a negative label.
- `uncertain`: visible but the boundary/point is ambiguous; retain an uncertainty flag and do not silently treat it as high-confidence ground truth.

A model must not be trained to interpret a default film-type setting as a confirmed negative. A fused femoral silhouette must never be converted into two invented centres. If two independent head centres cannot be supported, PI/PT reference eligibility and automated output must reflect that limitation; SS/LL may still be assessable.

## 5. Reader reference angles

Keep angle readings separate from model-training masks. Store reader 1 and reader 2 measurements independently, then store adjudicated values in a separate locked reference record. Record PI, PT, SS, LL, visibility/eligibility per parameter, and any adjudication note. Do not let a model prediction populate a reference field.

## 6. Annotation file format

The provisional JSON structure is defined by:

- `schemas/annotation.schema.json`
- `schemas/annotation.example.json`

Polygon coordinates are in original-image pixels. Each polygon must have at least three points. If a mask consists of multiple disconnected visible components, store multiple polygons. Preserve separate femoral-head polygons only where the reader can distinguish them. The union mask can be derived from the individually labelled heads or annotated as a union when only the combined region is defensible.

## 7. Quality control

- Review every annotation for swapped L1/S1 endpoints, reversed anterior/posterior orientation, malformed polygons, points outside the image, and implausible geometry.
- Double-annotate the pilot and locked test according to the approved SOP.
- Record reader code, protocol version, and uncertainty.
- Keep the original radiograph immutable; corrections create a new annotation version.
- Compute image hashes and keep an audit trail of annotation revisions.

## 8. Existing editor versus hand-drawn masks

The current Spine-Contour editor is useful for landmark corrections and femoral circle/rim-point placement. Its existing label record does not store arbitrary manually painted pixel masks for edited/manual vertebrae or full separate femoral-head polygons. Use the new research annotation format for hand-drawn masks unless and until a dedicated mask editor is implemented.
