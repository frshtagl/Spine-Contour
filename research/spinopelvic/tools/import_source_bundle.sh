#!/usr/bin/env bash
set -euo pipefail

# Import the user's source-only recovered-kit archive. It contains code/config/docs
# only: no weights, sample images, masks, annotations or case-level outputs.
EXPECTED_SHA256="aba797e420bbfc6bb98def2519d473690eadbacca3f2ade18d88ab37e0061629"
ARCHIVE=""
if [[ "$#" -ge 1 ]]; then ARCHIVE="$1"; fi
DEST="$(cd "$(dirname "$0")/../recovered_kits/source_bundle" && pwd)/contents"

if [[ -z "$ARCHIVE" || ! -f "$ARCHIVE" ]]; then
  echo "Usage: $0 /path/to/recovered_finetune_kits_source_only.tar.gz" >&2
  exit 2
fi

ACTUAL_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "Archive hash mismatch. Expected $EXPECTED_SHA256, got $ACTUAL_SHA256" >&2
  exit 3
fi

if tar -tzf "$ARCHIVE" | grep -E '(^/|(^|/)\.\.(/|$))' >/dev/null; then
  echo "Archive contains an unsafe path. Stop and inspect." >&2
  exit 5
fi

if [[ -e "$DEST" && -n "$(find "$DEST" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Destination is not empty: $DEST. Preserve it and choose a new clone/worktree rather than overwriting." >&2
  exit 4
fi
mkdir -p "$DEST"
tar -xzf "$ARCHIVE" --strip-components=1 -C "$DEST"

if find "$DEST" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.onnx' -o -name '*.dcm' -o -name '*.dicom' -o -name '*.nii' -o -name '*.nii.gz' -o -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.tif' -o -name '*.tiff' \) | grep -q .; then
  echo "Unexpected binary model/data artifact found in source-only archive. Stop and inspect." >&2
  exit 6
fi

echo "Imported verified source-only kit into: $DEST"
echo "No weights or case-level data were imported."
