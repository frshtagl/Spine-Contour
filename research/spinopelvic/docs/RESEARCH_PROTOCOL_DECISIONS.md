# Decisions to freeze before production annotation or training

This is a decision register, not an approved protocol. The investigator, radiology readers, supervisor and statistician must sign off the choices below before the 400-case split is frozen.

| Decision | Current working proposal | Required confirmation |
|---|---|---|
| Cohort allocation | 300 development = 240 train + 60 validation; 100 locked internal test | Confirm counts and sample-size rationale |
| Patient grouping | Patient-disjoint partitions; one examination per patient preferred | Confirm how repeat examinations are handled |
| Training labels | One adjudicated or protocol-selected reader record per development study | Define adjudication and tie-break rules |
| Test references | Two independent readers; adjudicated reference stored separately | Define measurement and point adjudication |
| Image hash | SHA-256 of exact original source file bytes | Keep consistent with the parent app's bundle-io hash method |
| DICOM decoding | Single-frame grayscale; slope/intercept and MONOCHROME1 inversion | Confirm acceptable image transfer syntax and modality policy |
| Orientation | Explicit anterior side; anatomical labels SA/SP are not screen-left/right | Approve reader instruction and mirrored-film handling |
| Vertebral masks | Separate L1–L5 polygons; all five levels reviewed before U-Net training | Define partial-level exclusion policy |
| Femoral target | One combined union mask plus two expert centres | Define fit QC and when PI/PT are withheld |
| S1 detector | Two S1 endpoints; recovered kit derives training box from keypoints plus padding | Decide whether to preserve source behavior or run a separately registered bbox-target experiment |
| Model route | U-Net segmentation and/or HRNet landmarks share S1/femoral components | Pre-register primary and secondary routes |
| Thresholds/QC | Freeze using development validation only | Define femoral fit acceptance, landmark confidence and withholding rules |
| Locked test | One confirmatory opening after all choices freeze | Define access authorization, analysis plan and reporting responsibilities |

## Non-negotiable leakage controls

- Patient-group codes must be present for every split-eligible study.
- No patient group may occur in both development and locked test.
- A split that cannot meet exact counts without dividing a patient group must fail rather than silently leak.
- Test images/labels never enter training or threshold selection.
- Missing/not_placed targets are not converted into negative labels.
- Every withheld angle has an explicit reason; no zero imputation.
- Record source checkpoint hash, Git commit, seed, split hash and environment for every run.
