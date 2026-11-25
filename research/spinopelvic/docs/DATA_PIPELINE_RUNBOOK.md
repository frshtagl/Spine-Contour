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
- geometry/reference_geometry_train.json and geometry/reference_geometry_validation.json, preserving the two expert femoral centres and S1 superior-endplate anchors separately from HRNet's 22 channels

It verifies source-file hashes and image dimensions. Raster images are copied byte-for-byte. Single-frame grayscale DICOM is decoded with the parent app's slope/intercept and MONOCHROME1 inversion, then saved as float TIFF for the PIL-based training kits; the annotation hash still verifies the original DICOM bytes.

It excludes U-Net examples with any L1–L5 status still not_placed; explicit not_visible is treated as background. The locked test set is excluded by default. The builder refuses to merge into a non-empty output directory and records a per-file output SHA-256 manifest plus an aggregate dataset fingerprint. It preserves original-image coordinates. The next step applies the parent app's exact S1 framing functions to create crop-frame inputs for U-Net, HRNet and femoral models; S1 detector training remains on unframed images because it is used to localize candidate windows at inference.

~~~bash
python research/spinopelvic/tools/make_source_frames.py \
  --dataset /secure/spinopelvic/model_inputs \
  --out /secure/spinopelvic/framed_model_inputs
~~~

The cropper calls backend.framing.reframe/fallback_window, saves raw crops (the model scripts then apply their own matching letterbox/preprocessing), rebases HRNet points into crop coordinates, crops vertebral/femoral masks and writes frame_manifest_all.json. This manifest is required to restore cropped predictions to original radiograph coordinates.

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

The locked-test output is written under a separate test_reference folder. Never pass it to a training command. After model selection is frozen, create its separate source-frame manifest:

~~~bash
python research/spinopelvic/tools/make_source_frames.py \
  --dataset /secure/spinopelvic/model_inputs_test_reference \
  --out /secure/spinopelvic/framed_model_inputs_test_reference \
  --include-test-reference
~~~

Use the resulting test-only frame manifest for locked-test prediction normalization; do not mix it with development/validation outputs.

Before using model predictions, run the synthetic framing checks with the parent backend dependencies installed:

~~~bash
python research/spinopelvic/tools/validate_end_to_end_framing.py
python research/spinopelvic/tools/validate_synthetic_s1_search.py
~~~

These scripts use generated synthetic arrays and a deterministic dummy S1 scorer. They check crop/restore geometry and the source locate/search/fixed-point selection path; they do not load the trained S1 detector or establish clinical framing accuracy.

## 5. Fine-tune the recovered model branches

The source-only bundle contains the recovered U-Net, HRNet and S1 training scripts. Example commands:

~~~
python research/spinopelvic/recovered_kits/source_bundle/contents/vertebra_unet_finetune/train.py \
  --img_dir /secure/spinopelvic/framed_model_inputs/vertebra/images \
  --mask_dir /secure/spinopelvic/framed_model_inputs/vertebra/masks \
  --val_names /secure/spinopelvic/framed_model_inputs/vertebra/val_names.txt \
  --ckpt backend/weights/vertebra_unet.pt \
  --out /secure/spinopelvic/runs/vertebra_unet

python research/spinopelvic/recovered_kits/source_bundle/contents/hrnet_landmarks_finetune/train.py \
  --ann /secure/spinopelvic/framed_model_inputs/hrnet/annotations_train.json \
  --val_ann /secure/spinopelvic/framed_model_inputs/hrnet/annotations_validation.json \
  --img_dir /secure/spinopelvic/framed_model_inputs/hrnet/images \
  --ckpt backend/weights/hrnet_landmarks.pt \
  --out /secure/spinopelvic/runs/hrnet

python research/spinopelvic/recovered_kits/source_bundle/contents/s1_finetune/train.py \
  --ann /secure/spinopelvic/model_inputs/s1/annotations_train.json \
  --val_ann /secure/spinopelvic/model_inputs/s1/annotations_validation.json \
  --img_dir /secure/spinopelvic/model_inputs/s1/images \
  --ckpt backend/weights/s1_keypointrcnn.pt \
  --out /secure/spinopelvic/runs/s1

python research/spinopelvic/tools/train_femoral_unet.py \
  --images /secure/spinopelvic/framed_model_inputs/femoral/images \
  --masks /secure/spinopelvic/framed_model_inputs/femoral/masks \
  --val-names /secure/spinopelvic/framed_model_inputs/femoral/val_names.txt \
  --checkpoint backend/weights/femoral_unet.pt \
  --out /secure/spinopelvic/runs/femoral \
  --seed 2026
~~~

These runs initialize from the original fork checkpoints. Do not chain one fine-tune into another unless that is a separately pre-registered experiment. The recovered U-Net/HRNet/S1 scripts contain historical training assumptions; compare the unchanged source baseline on the same validation cohort and report all new hyperparameters. The femoral script is a new research recipe, not the original hidden recipe. Its validation Dice/IoU are computed on positive-reference cases only; empty-reference specificity is reported separately so negative-only images cannot inflate the positive Dice score.

## 6. Normalize model predictions into separate model records

Initially create a separate coded CSV with reader_code MODEL for development/validation studies only, then generate empty templates. Keep locked-test MODEL templates separate until the model and thresholds are frozen. Do not pass a human-reader annotation folder as the base:

~~~bash
python research/spinopelvic/tools/make_annotation_templates.py \
  --images /secure/spinopelvic/images \
  --metadata-csv /secure/spinopelvic/model_template_manifest.csv \
  --out /secure/spinopelvic/annotations/MODEL
~~~

The normalizer deliberately clears human point/box/polygon fields and writes predictions into model_point, model_bbox and model_polygons.

The recovered HRNet and S1 predict.py scripts write predictions.json. The recovered vertebral U-Net predict.py writes one crop-frame label mask per image. HRNet/U-Net/femoral inference uses framed_model_inputs; S1 detector inference uses the original unframed model_inputs/s1 images. Example source-kit inference commands:

~~~bash
python research/spinopelvic/recovered_kits/source_bundle/contents/hrnet_landmarks_finetune/predict.py \
  --images /secure/spinopelvic/framed_model_inputs/hrnet/images \
  --ckpt /secure/spinopelvic/runs/hrnet/best.pt \
  --out /secure/spinopelvic/predictions/hrnet

python research/spinopelvic/recovered_kits/source_bundle/contents/s1_finetune/predict.py \
  --images /secure/spinopelvic/model_inputs/s1/images \
  --ckpt /secure/spinopelvic/runs/s1/best.pt \
  --out /secure/spinopelvic/predictions/s1

python research/spinopelvic/recovered_kits/source_bundle/contents/vertebra_unet_finetune/predict.py \
  --images /secure/spinopelvic/framed_model_inputs/hrnet/images \
  --ckpt /secure/spinopelvic/runs/vertebra_unet/best.pt \
  --out /secure/spinopelvic/predictions/vertebra
~~~

Generate femoral crop-frame masks for all framed images (not just the labelled training subset):

~~~bash
python research/spinopelvic/tools/predict_femoral_unet.py \
  --images /secure/spinopelvic/framed_model_inputs/hrnet/images \
  --checkpoint /secure/spinopelvic/runs/femoral/best.pt \
  --out /secure/spinopelvic/predictions/femoral
~~~

Normalize an HRNet route (S1 detector and femoral masks may be added as shared downstream components):

~~~bash
python research/spinopelvic/tools/normalize_model_predictions.py \
  --base-annotations /secure/spinopelvic/annotations/MODEL \
  --images /secure/spinopelvic/images \
  --hrnet-predictions /secure/spinopelvic/predictions/hrnet/predictions.json \
  --s1-predictions /secure/spinopelvic/predictions/s1/predictions.json \
  --femoral-mask-dir /secure/spinopelvic/predictions/femoral/masks \
  --frame-manifest /secure/spinopelvic/framed_model_inputs/frame_manifest_all.json \
  --model-name HRNet-S1-Femoral \
  --model-version experiment-01 \
  --component-manifest /secure/spinopelvic/checkpoint_hashes.json \
  --hrnet-raw-score-min 0.0 --s1-score-min 0.0 \
  --min-femoral-fit-confidence 0.0 --min-femoral-iou 0.0 \
  --out /secure/spinopelvic/predictions/normalized_hrnet
~~~

Normalize a separate vertebral U-Net route by replacing --hrnet-predictions with --vertebra-mask-dir /secure/spinopelvic/predictions/vertebra (the recovered predict.py writes *_mask.png files). Do not supply HRNet and vertebral U-Net together: both predict L1-L5 geometry and the normalizer intentionally rejects that combination. If HRNet and S1 predictions are both supplied, S1 detector points take precedence for S1; this precedence is recorded in the prediction manifest.

The femoral mask directory contains S1 crop-frame masks from predict_femoral_unet.py; the normalizer restores them to original radiograph coordinates using frame_manifest_all.json. The same manifest restores HRNet points and U-Net masks. Copy research/spinopelvic/configs/checkpoint_manifest.example.json to secure local experiment storage and fill in the exact baseline/selected checkpoint hashes, byte counts, preprocessing version and frozen thresholds. The component-manifest JSON should record the exact checkpoint SHA-256 values used for this run. The score thresholds above are placeholders only. Freeze them using development validation and the approved QC plan; HRNet raw heatmap peaks are not calibrated probabilities. The normalizer writes raw scores, accepted model points and rejected raw points into a secure prediction manifest. It never overwrites human-reviewed point/box/polygon fields in the base annotation files.

Validate the normalized prediction records:

~~~bash
python research/spinopelvic/tools/validate_annotations.py /secure/spinopelvic/predictions/normalized_hrnet
~~~

## 7. Geometry and agreement analysis


Compute source-compatible geometry from the normalized model prediction records (use /annotations/R1 and --reader-code R1 for human-reader geometry):

~~~
python research/spinopelvic/tools/compute_measurements_from_annotations.py \
  --annotations /secure/spinopelvic/predictions/normalized_hrnet \
  --reader-code MODEL \
  --min-model-score 0.0 \
  --out /secure/spinopelvic/measurements/model_source_geometry.csv
~~~

The tool calls the fork's existing backend.utils.spinopelvic_measurements_from_geometry implementation. For model predictions, use the normalized prediction annotation folder and its MODEL reader_code. The optional --min-model-score applies a single additional numeric threshold; prefer the per-component thresholds frozen in the normalizer because HRNet heatmap peaks, S1 detection scores and femoral fit confidence are on different scales.

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

## 8. Export selected custom checkpoints

After development-only selection:

~~~
python research/spinopelvic/tools/export_custom_checkpoint.py \
  --kind vertebra \
  --checkpoint /secure/spinopelvic/runs/vertebra_unet/best.pt \
  --output /secure/spinopelvic/onnx/vertebra_finetuned
~~~

Repeat for femoral, hrnet and s1. The exporter strict-loads the selected state dictionary, validates the ONNX graph and checks PyTorch/ONNX parity on synthetic random and zero inputs. This is graph validation, not real-image parity or clinical validation.

## 9. Locked-test rule

Open the 100-case locked test set once the model, thresholds, QC/withholding policy and analysis code are frozen. Report every attempted case, valid output, withheld output, exclusion and reason. Do not iterate on the test set; any post-test change requires a new independent evaluation cohort or a clearly labelled exploratory analysis.
