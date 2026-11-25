# Local research runbook — Spine-Contour spinopelvic

All commands below operate on local, approved data and write model artifacts outside Git. Replace example paths with the institution-approved secure locations. Do not place radiographs, DICOMs, annotation JSON, split manifests, reader references, trained weights, ONNX files or per-study logs in this public repository.

## 0. Environment and recovered source bundle

From the repository root:

~~~
python -m pip install -r research/spinopelvic/requirements-research.txt
~~~

The source-only recovered-kit archive is already available in the user's ChatGPT Library and local storage. It contains 27 source/configuration/documentation files, no weights and no case-level data. Import it into the research branch with:

~~~
bash research/spinopelvic/tools/import_source_bundle.sh /path/to/recovered_finetune_kits_source_only.tar.gz
~~~

The importer verifies SHA-256 and refuses to overwrite an existing extraction. The three original application checkpoints remain in their existing backend/weights paths. No weight upload or duplication is required.

## 1. Prepare coded annotation templates

Create a CSV in secure storage with columns:

study_id,patient_group_code,reader_code,stage,image_file,anterior_side

Optional paired columns: pixel_spacing_x_mm,pixel_spacing_y_mm.

- study_id and patient_group_code must be coded values, not direct identifiers.
- stage is one of pilot, development, or locked_test_reference.
- image_file is relative to the secure image directory.
- anterior_side is image_left, image_right, or unknown.
- Supported inputs: PNG, JPEG, BMP, TIFF and single-frame grayscale DICOM.

~~~
python research/spinopelvic/tools/make_annotation_templates.py \
  --images /secure/spinopelvic/images \
  --metadata-csv /secure/spinopelvic/annotation_manifest.csv \
  --out /secure/spinopelvic/annotations/R1
~~~

The tool hashes the exact original source file bytes. This matches the parent application's bundle-io.js sha256File implementation; it does not hash a re-encoded or resized image. For DICOM, it reads one grayscale frame, applies RescaleSlope/RescaleIntercept and MONOCHROME1 inversion exactly as the parent server decoder, and records PixelSpacing when available.

## 2. Annotate and validate

Use the approved SOP. Keep the five states distinct:

- segmented: accepted unchanged from model output.
- edited: corrected model output; preserve the original geometry.
- manual: manually placed after a model miss.
- not_placed: visible but not yet annotated; ignore, never negative.
- not_visible: explicitly checked and absent; negative where the target supports it.

Validate before splitting:

~~~
python research/spinopelvic/tools/validate_annotations.py /secure/spinopelvic/annotations/R1
~~~

The validator checks schema, image-coordinate bounds, polygon area, self-intersections and both corrected/model geometry. Do not resolve a schema failure by silently deleting a landmark or converting not_placed to not_visible.

## 3. Freeze patient-disjoint splits

After the cohort and reader protocol are approved:

~~~
python research/spinopelvic/tools/prepare_splits.py \
  --annotations /secure/spinopelvic/annotations \
  --train 240 --validation 60 --test 100 \
  --seed 2026 \
  --out /secure/spinopelvic/splits/frozen_split_manifest.json
~~~

The seed is an example only; replace it with the seed frozen in the approved protocol. The tool requires 300 unique development studies and 100 locked_test_reference studies, checks patient-group overlap, and fails if exact counts cannot be achieved without splitting a patient. The output manifest contains coded study IDs and must stay private. It refuses to overwrite an existing split unless --overwrite is explicitly supplied after a protocol amendment.

## 4. Build model-specific development inputs

Choose the adjudicated or otherwise protocol-defined training reader code. Do not select a convenient reader after looking at model results.

~~~
python research/spinopelvic/tools/build_training_inputs.py \
  --images /secure/spinopelvic/images \
  --annotations /secure/spinopelvic/annotations \
  --splits /secure/spinopelvic/splits/frozen_split_manifest.json \
  --reader-code ADJUDICATED \
  --out /secure/spinopelvic/model_inputs
~~~

The builder creates:

- vertebra/images + vertebra/masks + vertebra/val_names.txt
- femoral/images + femoral/masks + femoral/val_names.txt
- hrnet/images + train/validation 22-landmark JSON files
- s1/images + train/validation S1 keypoint JSON files

It verifies source-file hashes and image dimensions. Raster images are copied byte-for-byte. Single-frame grayscale DICOM is decoded with the parent app's slope/intercept and MONOCHROME1 inversion, then saved as float TIFF for the PIL-based training kits; the annotation hash still verifies the original DICOM bytes.

It excludes U-Net examples with any L1–L5 status still not_placed; explicit not_visible is treated as background. The locked test set is excluded by default. The builder refuses to merge into a non-empty output directory.

Only after the statistical analysis plan and model choices are frozen should the test references be prepared separately:

~~~
python research/spinopelvic/tools/build_training_inputs.py \
  --images /secure/spinopelvic/images \
  --annotations /secure/spinopelvic/annotations \
  --splits /secure/spinopelvic/splits/frozen_split_manifest.json \
  --reader-code R1 \
  --out /secure/spinopelvic/model_inputs_test_reference \
  --include-test-reference
~~~

The locked-test output is written under a separate test_reference folder. Never pass it to a training command.

## 5. Fine-tune the recovered model branches

The source-only bundle contains the recovered U-Net, HRNet and S1 training scripts. Example commands:

~~~
python research/spinopelvic/recovered_kits/source_bundle/contents/vertebra_unet_finetune/train.py \
  --img_dir /secure/spinopelvic/model_inputs/vertebra/images \
  --mask_dir /secure/spinopelvic/model_inputs/vertebra/masks \
  --val_names /secure/spinopelvic/model_inputs/vertebra/val_names.txt \
  --ckpt backend/weights/vertebra_unet.pt \
  --out /secure/spinopelvic/runs/vertebra_unet

python research/spinopelvic/recovered_kits/source_bundle/contents/hrnet_landmarks_finetune/train.py \
  --ann /secure/spinopelvic/model_inputs/hrnet/annotations_train.json \
  --val_ann /secure/spinopelvic/model_inputs/hrnet/annotations_validation.json \
  --img_dir /secure/spinopelvic/model_inputs/hrnet/images \
  --ckpt backend/weights/hrnet_landmarks.pt \
  --out /secure/spinopelvic/runs/hrnet

python research/spinopelvic/recovered_kits/source_bundle/contents/s1_finetune/train.py \
  --ann /secure/spinopelvic/model_inputs/s1/annotations_train.json \
  --val_ann /secure/spinopelvic/model_inputs/s1/annotations_validation.json \
  --img_dir /secure/spinopelvic/model_inputs/s1/images \
  --ckpt backend/weights/s1_keypointrcnn.pt \
  --out /secure/spinopelvic/runs/s1

python research/spinopelvic/tools/train_femoral_unet.py \
  --images /secure/spinopelvic/model_inputs/femoral/images \
  --masks /secure/spinopelvic/model_inputs/femoral/masks \
  --val-names /secure/spinopelvic/model_inputs/femoral/val_names.txt \
  --checkpoint backend/weights/femoral_unet.pt \
  --out /secure/spinopelvic/runs/femoral \
  --seed 2026
~~~

These runs initialize from the original fork checkpoints. Do not chain one fine-tune into another unless that is a separately pre-registered experiment. The recovered U-Net/HRNet/S1 scripts contain historical training assumptions; compare the unchanged source baseline on the same validation cohort and report all new hyperparameters. The femoral script is a new research recipe, not the original hidden recipe.

## 6. Geometry and agreement analysis

Compute source-compatible geometry from a selected reader's annotation records:

~~~
python research/spinopelvic/tools/compute_measurements_from_annotations.py \
  --annotations /secure/spinopelvic/annotations/R1 \
  --reader-code R1 \
  --out /secure/spinopelvic/measurements/R1_source_geometry.csv
~~~

The tool calls the fork's existing backend.utils.spinopelvic_measurements_from_geometry implementation. The same tool can process model-output records if they are converted to the research annotation schema with the model's points/masks and a model identifier used as reader_code.

Prepare a paired analysis CSV with columns:

study_id,patient_group_code,model_id,parameter,prediction_deg,reference_deg,withheld_reason

Use one row per model × study × parameter. Missing/withheld predictions remain blank; never replace them with zero.

~~~
python research/spinopelvic/analysis/analyze_agreement.py \
  --input /secure/spinopelvic/analysis/paired_predictions.csv \
  --out /secure/spinopelvic/analysis/agreement_report.json \
  --bootstrap 2000 --seed 2026
~~~

The report includes coverage, MAE, median absolute error, bias, RMSE, Bland–Altman limits, ICC(A,1), and patient-cluster bootstrap intervals. Interpret ICC alongside error and coverage; it is not a substitute for clinical review.

## 7. Export selected custom checkpoints

After development-only selection:

~~~
python research/spinopelvic/tools/export_custom_checkpoint.py \
  --kind vertebra \
  --checkpoint /secure/spinopelvic/runs/vertebra_unet/best.pt \
  --output /secure/spinopelvic/onnx/vertebra_finetuned
~~~

Repeat for femoral, hrnet and s1. The exporter strict-loads the selected state dictionary, validates the ONNX graph and checks PyTorch/ONNX parity on synthetic random and zero inputs. This is graph validation, not real-image parity or clinical validation.

## 8. Locked-test rule

Open the 100-case locked test set once the model, thresholds, QC/withholding policy and analysis code are frozen. Report every attempted case, valid output, withheld output, exclusion and reason. Do not iterate on the test set; any post-test change requires a new independent evaluation cohort or a clearly labelled exploratory analysis.
