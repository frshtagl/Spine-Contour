# Research data governance and public-repository boundary

## Public Git repository contents

This repository is a public fork. Keep it limited to:

- source code and configuration;
- annotation schemas and synthetic examples;
- synthetic tests and CI workflows;
- de-identified, non-study documentation;
- aggregate results only after institutional review confirms they are safe to publish;
- checkpoint hashes and provenance metadata that do not expose restricted artifacts.

Do not commit radiographs, DICOMs, masks, polygons, point annotations, patient-group manifests, reader measurements, per-study predictions, training logs containing case identifiers, trained checkpoint binaries, ONNX exports or other case-level artifacts.

## De-identification is not a public-release decision

De-identification reduces direct identifiers but does not automatically eliminate re-identification risk. Lateral radiographs may contain embedded annotations, burned-in identifiers, unusual anatomy or metadata that can be linkable. Coded study IDs and patient-group codes are still sensitive research data when the re-identification key or linkage information exists.

Keep the re-identification key separate from research files, restrict access to the coded dataset, encrypt storage and transfers, and follow the approved retention/deletion policy. The institution's ethics, data-access and publication approvals govern whether data may be processed locally, shared across sites or sent to cloud compute.

## Locked test separation

- Keep test images, reference annotations and measurement tables in a separate access-controlled location.
- Do not expose test labels to training scripts or model-selection code.
- The dataset builder excludes locked test by default. Its explicit --include-test-reference option writes test data into a separate directory; that output is never a training input.
- Keep an audit trail of access and freeze the analysis plan before test evaluation.

## Cloud/Colab

Use Colab or another external service only if written institutional and site-specific approval explicitly permits the proposed data transfer and processing. If not approved, use institution-controlled compute. This repository's public status does not change those obligations.

## Model artifacts

The original fork checkpoints remain in their existing locations and are not duplicated by this research branch. Fine-tuned checkpoints and ONNX exports belong in approved private model-artifact storage. The public repository may contain model cards, training code, hashes and approved aggregate evaluation summaries, not the model binaries unless their publication is separately authorized.
