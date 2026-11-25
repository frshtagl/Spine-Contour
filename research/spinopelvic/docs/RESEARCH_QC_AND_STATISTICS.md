# QC, withholding and statistical reporting

## Geometry implementation

Spinopelvic angles are computed through the fork's existing backend.utils.spinopelvic_measurements_from_geometry function. The research utility maps named anatomical endpoints into that function rather than introducing a second independent formula implementation.

- SS: sacral slope from the S1 superior endplate.
- PI and PT: require both S1 endpoints and two femoral-head centres.
- LL_L1_S1: requires the L1 and S1 superior endplates.
- Other partial lumbar levels may produce the corresponding level-to-S1 lordosis values.

The two expert femoral-head centres are used to determine the hip midpoint. The geometry adapter supplies a positive radius placeholder only because the source function's input contract expects circle triples; that placeholder is not an anatomical radius and is not reported.

## QC and withholding

- Preserve the source annotation states. not_placed is incomplete, not a negative label.
- Do not generate PI/PT when the two femoral centres are not independently supportable or the femoral fit fails QC.
- Do not invent a missing S1 or L1 endplate.
- Withhold only the parameter(s) whose required inputs fail QC; retain independent valid parameters where supported.
- Record a reason for every withheld output.
- Report coverage with accuracy. A model must not appear better because it abstains on difficult cases.
- Never impute a withheld angle as zero.

## Required model metrics

- Vertebral U-Net: per-level Dice, foreground mean Dice, boundary/contour review and class-presence failures.
- Femoral U-Net: positive-reference union-mask Dice/IoU, empty-reference specificity, edge truncation, connected-component behavior and two-disc-fit acceptance. Empty-reference cases are reported separately, not counted as perfect Dice.
- HRNet: per-landmark Euclidean localization error in original-image pixels, per-landmark P95, missing-point rate and left-/right-facing subgroup results. State pixel units and image dimensions; do not compare historical val_px without establishing its coordinate frame.
- S1 detector: SA/SP localization error, endplate-angle error, detection misses, endpoint swaps and facing-stratified results.
- End-to-end: run with predicted S1 framing/crop, not only ground-truth-framed crops.

## Clinical measurement agreement

For PI, PT, SS and L1–S1 LL, report:

- number of eligible references, number predicted, number withheld, and coverage;
- MAE, median absolute error, signed bias and RMSE;
- Bland–Altman bias and 95% limits of agreement;
- ICC(A,1) with a confidence interval;
- patient-cluster bootstrap 95% confidence intervals for MAE, bias and ICC;
- paired Model A vs Model B differences on the same eligible patients;
- coverage differences and reasons for missing outputs.

The analysis helper implements these summaries for paired CSV rows. The bootstrap resamples coded patient groups, not individual examinations, so repeated studies from one patient stay clustered. The ICC is a two-way random-effects, absolute-agreement, single-measure ICC; do not report it alone.

## Test discipline

The 100-case locked test set is used once after model, thresholds, QC rules and analysis are frozen. Any threshold change, model change, preprocessing change or code adjustment prompted by test results invalidates the original test as a final confirmatory estimate. Label any subsequent analysis exploratory or obtain a new independent test cohort.
