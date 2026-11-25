# Google Colab workflow — planned

This folder records the intended Colab workflow. It is not an executable training notebook yet; the dataset builder and trainer have not been implemented.

## Non-negotiable data approval

The uploaded UHS synopsis says that any cross-institutional data movement or external/cloud computation requires site-specific approval. Do not upload any radiograph, DICOM, mask, annotation JSON, reference reading, or patient-linked manifest to Google Colab/Drive until written UHS and relevant hospital approvals explicitly authorize that exact processing route.

The GitHub repository is public. It is for code, documentation, synthetic examples, and approved aggregate metrics only.

## Intended setup after approval

1. Open a Colab notebook using an institution-approved account and storage location.
2. Select a GPU runtime and record GPU model, CUDA, Python, PyTorch, and package versions.
3. Clone the research branch:

   ```bash
   git clone --branch research/spinopelvic-lab https://github.com/frshtagl/Spine-Contour.git
   cd Spine-Contour
   ```

4. Install pinned training dependencies from the environment file that will be added with the trainer.
5. Mount/read the approved dataset from the institution-authorized storage location. Do not place it inside the cloned Git repository.
6. Run annotation validation and patient-level split checks before training.
7. Train using only the 240 training cases; use the 60 validation cases for selection.
8. Save checkpoints and detailed logs to the approved secure storage, not the public GitHub repository.
9. Evaluate the locked 100-case test only once the analysis plan and model are frozen.
10. Commit code, YAML configs, synthetic tests, and approved aggregate metrics to the research branch. Do not commit patient data or weights by default.

## Persistence

Colab runtimes are ephemeral. Checkpoints and logs must be saved to approved persistent storage during training. A Git commit is not a substitute for a secure model-artifact store.

## Pending implementation

- Reproducible environment/requirements file
- Dataset builder and schema validator
- Patient-level split generator
- Training entry points for segmentation and HRNet
- Checkpoint/resume support
- Evaluation and report scripts
- Colab notebook with preflight checks and run configuration
