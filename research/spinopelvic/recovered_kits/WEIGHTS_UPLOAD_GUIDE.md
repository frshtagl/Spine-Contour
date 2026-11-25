# Private checkpoint and source import guide

## Gate 0 — change repository visibility

The repository was verified as **PUBLIC** on 25 November 2026. The connected GitHub integration exposes repository content writes but no repository-visibility mutation action.

Open `https://github.com/frshtagl/Spine-Contour/settings`, then:

**General → Danger Zone → Change repository visibility → Make private**.

Verify GitHub displays the Private label. Existing public forks, prior clones and downloads do not become private retroactively.

## Gate 1 — pull the prepared research branch

```bash
git switch research/spinopelvic-lab
git pull origin research/spinopelvic-lab
```

## Gate 2 — enable Git LFS

```bash
git lfs install
git lfs ls-files
```

The root `.gitattributes` now contains explicit LFS rules for recovered `.pt` and `.pth` files. The research `.gitignore` allows only the named recovered checkpoint directories; other weights and all case-level data remain ignored.

## Gate 3 — restore only the three checkpoint files

Extract the source-only archive into a local working directory. Its `provenance/restore_weights.py` extracts only the checkpoint members from the original ZIPs and verifies each SHA-256 and byte count. It does not extract sample radiographs, masks or annotation JSON.

Run from the extracted source-only folder:

```bash
python provenance/restore_weights.py \
  --unet-zip "/path/to/files (3).zip" \
  --hrnet-zip "/path/to/files (2).zip" \
  --s1-zip "/path/to/s1_finetune_kit.zip" \
  --output-root "/path/to/Spine-Contour/research/spinopelvic/recovered_kits"
```

Expected output paths:

- `recovered_kits/vertebra_unet_finetune/weights/vertebra_unet.pt`
- `recovered_kits/hrnet_landmarks_finetune/weights/landmarks_hrnet.pt`
- `recovered_kits/s1_finetune/weights/s1_keypointrcnn.pt`

Stop if any hash or size differs.

## Gate 4 — commit weights through LFS

Only after the repository is private:

```bash
git add .gitattributes research/spinopelvic/recovered_kits
git lfs ls-files
git status --short
git commit -m "Add recovered fine-tuning checkpoints via Git LFS"
git push origin research/spinopelvic-lab
```

Check that the three recovered checkpoint paths appear in `git lfs ls-files` and that the push completes. Do not force-push.

## Keep these files out of Git, even when private

- Patient radiographs / DICOMs
- Masks and pseudo-masks
- Annotation JSON with patient-linked coordinates
- Patient manifests and train/test split manifests containing identifiers
- Overlays, prediction files, measurements and training outputs

Use institution-approved encrypted storage for the actual 300 annotated development cases and 100 locked test cases. The Git repository should hold code, schemas, configs, aggregate evaluation summaries and model artifacts whose storage/redistribution is permitted.

## Important baseline distinction

The supplied U-Net checkpoint matches the existing fork's LFS pointer. The supplied HRNet and S1 checkpoint hashes differ from the fork's current LFS pointers. Keep the recovered kit files under `recovered_kits/`; do not overwrite `backend/weights/`.
