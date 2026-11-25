# Exact annotation strategy extracted from Spine-Contour

**Repository:** frshtagl/Spine-Contour  
**Source branch:** main  
**Source commit inspected:** 5df84e0af578e8faabab30b24e655735c3663ad4  
**Audit date:** 25 November 2026

This document describes what the repository actually implements. It separates the application annotation method from the research-specific pixel-mask schema and from future retraining proposals.

## 1. What the application annotator records

Spine-Contour's level editor is a geometric landmark/circle editor. It is not a general-purpose manual pixel-mask painting tool.

For a lumbar/lumbopelvic lateral radiograph, the relevant targets are:

| Structure | Geometry stored | Exact interpretation |
|---|---|---|
| L1–L5, each | SA, SP, IA, IP | Superior-anterior, superior-posterior, inferior-anterior and inferior-posterior vertebral-body corners |
| S1 | SA, SP | Anterior and posterior endpoints of the S1 superior endplate |
| Femoral head 1 | Centre, radius, optional rim | Circle fitted from 2–4 points on the femoral-head rim |
| Femoral head 2 | Centre, radius, optional rim | A separate circle fitted from 2–4 rim points |

The lumbar film-type defaults show the lumbar, sacral and femoral-head groups. The other editor groups remain in the record but normally start not visible.

### 1.1 Lumbar vertebral bodies

The standard placement sequence is SA → SP → IA → IP. The point-step chips allow the annotator to choose a different next point or re-place a point.

A partially visible body may be finished with one complete endplate pair only:

- SA + SP, or
- IA + IP.

A single isolated corner is not a valid partial placement. The missing endplate pair is represented by two null points, not by an invented location.

If the model geometry is correct, it is accepted unchanged. If a corner is moved, the entry becomes edited. If the model did not place a visible body and the annotator places it, the entry becomes manual.

### 1.2 S1 superior endplate

S1 is represented by its two superior-endplate endpoints, anterior and posterior. The labels are anatomical, not simply left/right screen coordinates. This matters when the displayed radiograph is mirrored.

### 1.3 Femoral heads

The two femoral heads are annotated independently. Two rim points are interpreted as a diameter; three or four points are fitted to a circle. The stored geometry is centre and radius, with the rim points retained when supplied.

The application does not store a manually traced femoral-head pixel mask in this editor. The inference pipeline has a separate combined femoral-region U-Net output and a two-disc fitting step.

## 2. Five exact annotation states

| State | Meaning | Training interpretation |
|---|---|---|
| segmented | Visible; geometry equals model output | Positive, human-accepted model result |
| edited | Visible; geometry differs from model output | Positive corrected result; retain original model geometry for comparison |
| manual | Visible; model had no geometry and annotator placed it | Positive label and model miss |
| not_placed | Visible but no geometry has been placed | Incomplete/ignored; never treat as a negative |
| not_visible | Annotator has marked the structure absent from the film | Explicit negative where the model target supports it |

The source field records whether the entry came from model output, annotator action or film-type defaults. The distinction between not_placed and not_visible is essential: treating an unreviewed visible level as absent creates a false negative.

## 3. Label-record format

The application writes one record per film at:

    <userData>/labels/<stem>.label.json

The schema identifier is spinecontour.segmentation-label/1. The record includes:

- study filename stem;
- image filename, SHA-256 of the exact original source file bytes, width and height. This was verified in bundle-io.js: sha256File streams the original file bytes into SHA-256; it is not a hash of decoded grey pixels or a re-encoded image.
- film region and view;
- model name and model version/hash;
- application version and save timestamp;
- completeness;
- levels keyed by anatomical code;
- two femoral-head entries;
- edited-point counts and optional comment/flags.

Coordinates are stored in pixels of the decoded image, with origin at the top-left. They are not normalized coordinates. The image fingerprint is a separate hash of original file bytes. For DICOM, backend/server.py::_dicom_pixel_array reads one grayscale frame, applies RescaleSlope/RescaleIntercept and MONOCHROME1 inversion; the source image bytes are still what get hashed.

For a body, corners are stored as SA, SP, IA and IP. A segmented body may retain the model's mask outline as a contour. Edited and manual bodies have no model contour in the record; their four-corner geometry is not a hand-traced pixel mask.

The edited_points list records which geometry keys changed. For a manual entry it records the keys the annotator placed. The top-level edits summary counts segmented, edited, manual, not_placed and not_visible entries, plus moved and placed points.

## 4. Application review workflow

1. Open the correct real film and verify its region/view.
2. Inspect the model result, if one exists.
3. Enter Edit segmentation.
4. Review visibility for each relevant level. Keep visible-but-unplaced distinct from not visible.
5. Review L1–L5: accept, correct or manually place the four corners.
6. Review the two S1 superior-endplate endpoints.
7. Review both femoral-head circles independently.
8. Add a film-level comment where the next reader needs context.
9. Save edits to write the label record.
10. Confirm no visible target remains not_placed unless its non-assessability is explicitly documented under the research protocol.

Relevant controls include UNDO, CANCEL, FINISH for a complete partial endplate pair, FIT for a femoral circle, CLEAR for a manual level, REVERT for one edited level, RESET TO PREDICTION for the full model geometry, Discard for the saved baseline, and Save edits for the record.

Window/level is display-only and is not part of the label record. The editor supports high zoom and keyboard nudging for point correction.

## 5. Compatibility with the model architecture

| Model branch | Source architecture/output | Relationship to annotation |
|---|---|---|
| Vertebral U-Net | ResNet-34 U-Net, one grayscale input, six classes including background | The application has model mask contours for unchanged segmented bodies; edited/manual records supply corners, not manually traced masks |
| HRNet | HRNet-W32, 22 heatmaps: four corners for L1–L5 plus two S1 points | Direct match to the 22 named vertebral/S1 landmarks |
| S1 detector | ResNet-50 FPN Keypoint R-CNN with two keypoints | Direct match to the two S1 endpoints; research schema additionally stores a box |
| Femoral U-Net | ResNet-34 U-Net with one combined foreground channel | Research-specific femoral union polygons can provide a target; the application editor itself stores two circles, not that union mask |

The source retraining planning note explicitly recognises the vertebral U-Net mismatch: its target requires masks, while edited/manual editor records contain four corners. The note lists possible future directions, including prioritising HRNet landmarks or rasterising quadrilaterals. These are design options, not an already implemented training pipeline.

## 6. What the repository does not provide

The public repository does not provide its original lumbar training dataset or a complete original lumbar training recipe. The file backend/models/training.py builds checkpoint architectures for export/validation; it is not a training loop.

The annotation editor documentation does not define a full clinical research SOP, two-reader adjudication protocol, inter-reader reliability plan, or final PI/PT/SS/LL statistical analysis.

## 7. Source files checked

- docs/level-editor.md
- docs/design/level-editor/label-record.schema.json
- docs/design/level-editor/label-record.example.json
- docs/design/level-editor/label-record.partial.example.json
- renderer/data/levels.js
- renderer/data/label-record.js
- backend/models/training.py
- backend/models/hrnet.py
- backend/models/models.py
- backend/femoral.py
- docs/superpowers/NEXT-SESSION-retraining.md
- docs/superpowers/specs/2026-10-15-annotation-bundle-export-import-design.md

This is a source-derived extraction, not a claim that the research-specific pixel-mask schema is part of the original application editor.


## 8. Hashing and DICOM decoding correction (verified from source)

The initial draft in this document described the image SHA-256 as a hash of stored grey pixels. That wording was incorrect. The application bundle export/import code calls sha256File on the original file and hashes its raw bytes. The research schema and template generator now follow that exact behavior.

For DICOM, the parent backend decodes a single grayscale pixel array, applies RescaleSlope and RescaleIntercept, and inverts MONOCHROME1. The research template and dataset tools mirror this decoding path. The DICOM file hash remains over the original file bytes; model-input TIFFs are derived copies and do not replace the source fingerprint.
