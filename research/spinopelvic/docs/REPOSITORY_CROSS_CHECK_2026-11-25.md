# Research branch cross-check — 25 November 2026

## Repository identity and branch verification

The personal fork is frshtagl/Spine-Contour, forked from Feches/Spine-Contour.

The research work is on branch:

    research/spinopelvic-lab

It is not on main. The branch was checked through GitHub's branch ref, commit and recursive tree endpoints. At the time of this audit:

- branch head: dd9e2c206cdacf16ee435f17d08f3e6bc9c2e022;
- branch comparison: 10 commits ahead of main;
- research/spinopelvic directory: 17 tracked files;
- the research directory is therefore not empty on this branch.

If the repository is opened on main, the research folder will not appear. Open the branch URL instead:

https://github.com/frshtagl/Spine-Contour/tree/research/spinopelvic-lab/research/spinopelvic

## What is actually present

The 17-file research workspace contains:

- README, STATUS, baseline inventory and annotation protocol;
- training plan and synopsis alignment/amendment notes;
- upstream/license note;
- research configuration and requirements;
- research annotation JSON schema and examples;
- schema-validation tests;
- checkpoint-inspection and annotation-validation utilities;
- a Colab usage README and private-data ignore rules.

The upstream application files and annotation schema were also inspected. The exact annotation method is documented separately in ANNOTATION_STRATEGY_EXACT.md.

## Deficiencies confirmed

The branch is a research scaffold, not a complete executable training project. The following implementation components were absent at the audited head:

| Component | Status |
|---|---|
| Patient-group-disjoint split generator | Missing |
| Image/annotation dataset builder | Missing |
| Model-specific training loops | Missing |
| Validation and locked-test evaluation pipeline | Missing |
| Fine-tuned checkpoint ONNX exporter | Missing |
| Executable Colab notebook | Missing |
| End-to-end evaluation using predicted S1 framing | Not established |
| PI/PT/SS/LL agreement, Bland–Altman, ICC and confidence-interval analysis | Missing |

The source backend/models/training.py is an architecture/checkpoint builder for export and validation. It must not be described as an existing training implementation.

The research annotation schema is a study-specific extension. It must not be attributed to the original application editor. In particular, the editor's four vertebral corners are not equivalent to a hand-traced vertebral mask.

## Verification boundary

This audit verified repository identity, branch ref, commit ancestry, tracked file tree and the contents of the relevant source/documentation files.

It did not run a real model training job, strict-load the original checkpoint tensor files, process patient radiographs, or produce clinical performance results. Git LFS checkpoint availability and numerical parity must be checked in a clone after Git LFS retrieval.

## Required next implementation sequence

1. Freeze the study-level annotation SOP, reader roles, adjudication rules and anatomical orientation convention.
2. Implement and test a patient-level split utility; ensure all images from one patient remain in one partition.
3. Implement image-hash verification and a model-specific dataset builder. Keep patient manifests, images, masks and annotations outside Git.
4. Resolve how edited/manual vertebral corners become U-Net masks, or make the HRNet landmark route the primary model.
5. Implement model-specific fine-tuning from the original checkpoints with reproducible configuration and checkpoint provenance.
6. Implement validation-only model selection and separate locked-test evaluation.
7. Evaluate the complete inference path, including predicted S1 localization/cropping; ground-truth-framed crops alone are not end-to-end evaluation.
8. Implement the final PI/PT/SS/LL agreement analysis against independent reader references and adjudicated measurements.
9. Export selected checkpoints to ONNX, verify numerical parity, and retain model hashes.

## Privacy boundary

Only source code, synthetic tests and approved non-identifying documentation belong in this public fork. Radiographs, DICOMs, patient-linked manifests, annotations, reader measurements, NPZ datasets, training logs, checkpoints and exported ONNX files must remain in approved private storage. Colab/cloud processing requires explicit written institutional and site approval.


## Follow-up implementation update — 25 November 2026

The table above is a historical snapshot of the branch at the original audit commit. Subsequent commits added:

- patient-disjoint split generation and split-manifest schema;
- annotation template generation and stronger polygon/coordinate validation;
- image-hash-verified dataset building for vertebral U-Net, femoral U-Net, HRNet and S1;
- single-frame grayscale DICOM decoding aligned to the parent server;
- source-compatible geometry extraction and clustered-bootstrap agreement analysis;
- femoral U-Net fine-tuning and custom-checkpoint ONNX export utilities;
- synthetic tests, reference-measurement schema, experiment templates, model cards and branch CI.

Still absent: imported source-only U-Net/HRNet/S1 training scripts in the branch tree (the verified archive remains in the user's Library/local PC), end-to-end S1 framing/coordinate tests, prediction-output normalization, Colab notebook and all real-data/checkpoint/training validation. No weights or study data were uploaded in the follow-up.
