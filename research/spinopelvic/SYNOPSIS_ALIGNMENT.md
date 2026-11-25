# Alignment with the tentative UHS synopsis

## Study question and endpoints

The uploaded synopsis proposes a reliability-aware deep-learning framework for PI, PT, SS, and L1–S1 LL on standing lateral lumbopelvic radiographs. It emphasizes that an angle should be withheld when the anatomical evidence needed for that specific angle is inadequate.

The research software must keep these output dependencies separate:

- SS: S1 superior-endplate line.
- LL: L1 and S1 superior-endplate lines.
- PI and PT: S1 geometry plus a bicoxofemoral midpoint supported by two independently identifiable femoral-head centres.

A merged pelvic silhouette or a single fused femoral mask centroid is not an acceptable substitute for the midpoint of two independently supported head centres.

## Two-method design

### Model A — segmentation-to-geometry

The synopsis names 2D nnU-Net v2 as the primary segmentation approach. The proposed targets are L1–L5 vertebral anatomy, the S1 superior-endplate region, and the femoral-head region. Explicit line fitting and centre extraction follow segmentation.

### Model B — direct landmark heatmaps

The synopsis names a heatmap-based direct-landmark comparator with six mandatory points: two L1 superior-endplate endpoints, two S1 superior-endplate endpoints, and two independent femoral-head centres.

Both methods must share the same image partitions, frozen geometry calculator, QC rules, and locked test references.

## Updated working sample plan

The synopsis currently says approximately 420 examinations: 300 development and 120 locked test. The investigator's current working plan is 400 total:

| Partition | n | Notes |
|---|---:|---|
| Train | 240 | Training only |
| Validation | 60 | Architecture/hyperparameter/QC-threshold selection |
| Locked internal test | 100 | No tuning or iterative feedback |
| **Total** | **400** | **300 development + 100 test** |

The 100-case test size is a working choice, not a completed statistical justification. The synopsis illustrates a precision calculation using SD = 3° and a 95% CI half-width of 0.55°, which gives approximately 115 independent test examinations. With n = 100 and the same illustrative SD, the approximate half-width is 1.96 × 3 / √100 = 0.588°. Revisit this with the statistician and supervisor before the test set is locked. Do not present 100 as definitively powered without that review.

Use patient-level separation. Prefer one examination per patient. Stratify by site where feasible. If repeated examinations are unavoidable, keep all examinations from one patient in one partition and adjust the analysis accordingly.

The 30–50-image annotation/repeat-reading pilot described in the synopsis must not include locked-test patients. It may be part of the development cohort only if the protocol and partition manifest make that explicit.

## Reference standard

- Two trained clinical readers independently annotate the test references while blinded to model predictions.
- Freeze the landmark definitions, angle signs, visibility rules, and adjudication threshold after the pilot.
- The synopsis lists >5° as an initial candidate adjudication trigger, not a final approved threshold.
- Keep reader 1, reader 2, and adjudicated reference values separately.
- Do not use AI masks as the test ground truth.

## Outcomes to preserve

Primary:
- Parameter-specific MAE in degrees for PI, PT, SS, and LL, with 95% confidence intervals.
- Proportion of reference-eligible test examinations receiving all four valid outputs.

Secondary:
- Per-parameter coverage and withholding reasons.
- Signed bias and Bland–Altman limits of agreement.
- Absolute-agreement ICC(A,1), with confidence intervals.
- Paired model differences on the same patients where both methods return a valid result.
- Landmark localization error; segmentation Dice; line-angle error; human-reader variability; runtime.

Missing or withheld values must never be replaced with zero. Report the number screened, excluded, reference-ineligible, attempted, accepted, and withheld.

## Items to reconcile before final synopsis submission

1. Replace the tentative 420/300/120 count if the final approved plan is 400/240/60/100, and update the sample-size precision justification.
2. Decide whether Model A remains nnU-Net v2 or whether a compatible Spine-Contour ResNet-34 U-Net warm-start is also included as a separate segmentation experiment. These are not the same architecture.
3. Define how two overlapping femoral heads are annotated and when two independent centres are considered supportable.
4. Define whether the 300 development cases include the 30–50-image reader-reliability pilot.
5. Confirm the exact image orientation, pixel calibration, and coordinate conventions in the annotation SOP.
6. Obtain written approval for any cross-site transfer or cloud/Colab processing before data are uploaded.
7. Confirm that the locked 100-case test is sufficiently precise for the intended primary estimates.

This file records technical alignment; it does not replace supervisor, statistician, radiology-reader, or ethics approval.
