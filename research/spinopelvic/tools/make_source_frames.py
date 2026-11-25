#!/usr/bin/env python3
"""Create source-app S1-framed training inputs for vertebra/HRNet/femoral models.

S1 detector training remains on the unframed images because the app uses it to
search candidate windows. U-Net, HRNet and femoral models are trained on the
S1-framed crop used by backend.models.models.spinopelvic_prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from backend import framing  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Expected a grayscale numeric radiograph: {path}")
    return array


def crop_window(record: dict[str, Any], image: np.ndarray) -> tuple[tuple[int, int, int, int], str]:
    s1 = record.get("s1_endplate") or {}
    sa, sp = s1.get("SA"), s1.get("SP")
    if sa is not None and sp is not None:
        proposed = framing.reframe(np.asarray([sa, sp], dtype=np.float64), image.shape)
        if proposed is not None:
            return tuple(int(v) for v in proposed), "s1_reframe"
    return tuple(int(v) for v in framing.fallback_window(image)), "fallback_window"


def save_tiff(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array)).save(path, format="TIFF")


def translate_point(point: Any, window: tuple[int, int, int, int]) -> list[float] | None:
    if point is None:
        return None
    left, top, right, bottom = window
    x, y = float(point[0]), float(point[1])
    if not (left <= x < right and top <= y < bottom):
        return None
    return [x - left, y - top]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Output from build_training_inputs.py")
    parser.add_argument("--out", type=Path, required=True, help="New secure output folder outside Git")
    parser.add_argument("--include-test-reference", action="store_true")
    parser.add_argument("--allow-in-repo", action="store_true", help="Only for synthetic tests; never use with real data")
    args = parser.parse_args()
    try:
        if not args.allow_in_repo:
            try:
                args.out.expanduser().resolve().relative_to(ROOT.resolve())
            except ValueError:
                pass
            else:
                raise ValueError("Refusing to write framed case-level datasets inside Git")
        if args.out.exists() and any(args.out.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {args.out}")
        args.out.mkdir(parents=True, exist_ok=True)
        splits = ["train", "validation"] + (["test_reference"] if args.include_test_reference else [])
        all_frames = []
        validation_names = {"vertebra": [], "femoral": []}
        for split in splits:
            base = args.dataset / "test_reference" if split == "test_reference" else args.dataset
            reference_path = base / "geometry" / f"reference_geometry_{split}.json"
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            frame_by_original_file = {}
            frame_rows = []
            for study in reference["studies"]:
                original_file = study["file"]
                source = base / "hrnet" / "images" / original_file
                if not source.is_file():
                    raise FileNotFoundError(f"Missing source image for frame: {source}")
                image = read_gray(source)
                height, width = image.shape
                if study.get("image_sha256") is None:
                    raise ValueError(f"Missing source image hash for {study['study_id']}")
                window, method = crop_window(study, image)
                left, top, right, bottom = window
                if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                    raise ValueError(f"Invalid source frame window for {study['study_id']}: {window}")
                crop = image[top:bottom, left:right]
                if crop.size == 0:
                    raise ValueError(f"Empty framed crop for {study['study_id']}")
                framed_file = f"{Path(original_file).stem}.tiff"
                framed_image_path = args.out / "hrnet" / "images" / framed_file
                save_tiff(crop, framed_image_path)
                frame = {
                    "study_id": study["study_id"],
                    "patient_group_code": study["patient_group_code"],
                    "image_sha256": study["image_sha256"],
                    "original_file": original_file,
                    "s1_file": original_file,
                    "framed_file": framed_file,
                    "framed_file_sha256": sha256_file(framed_image_path),
                    "task_mask_sha256": {},
                    "original_width_px": width,
                    "original_height_px": height,
                    "window": [left, top, right, bottom],
                    "crop_width_px": right - left,
                    "crop_height_px": bottom - top,
                    "framing_method": method,
                    "s1_endplate_original": study.get("s1_endplate"),
                    "anterior_side": study.get("anterior_side", "unknown"),
                }
                frame_rows.append(frame)
                all_frames.append(frame)
                frame_by_original_file[original_file] = frame

            ann_path = base / "hrnet" / f"annotations_{split}.json"
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
            framed_items = []
            for item in ann.get("images", []):
                frame = frame_by_original_file[item["file"]]
                window = tuple(frame["window"])
                points = {name: translate_point(point, window) for name, point in item.get("landmarks", {}).items()}
                framed_items.append({"file": frame["framed_file"], "landmarks": points})
            out_base = args.out
            (out_base / "hrnet").mkdir(parents=True, exist_ok=True)
            (out_base / "hrnet" / f"annotations_{split}.json").write_text(json.dumps({"images": framed_items}, indent=2) + "\n", encoding="utf-8")

            for task in ("vertebra", "femoral"):
                src_images = base / task / "images"
                src_masks = base / task / "masks"
                for source in sorted(src_images.glob("*")):
                    if not source.is_file():
                        continue
                    frame = frame_by_original_file.get(source.name)
                    if frame is None:
                        raise ValueError(f"No frame metadata for {task} image {source.name}")
                    window = tuple(frame["window"])
                    left, top, right, bottom = window
                    image = read_gray(source)
                    if image.shape != (frame["original_height_px"], frame["original_width_px"]):
                        raise ValueError(f"Image size changed before framing: {source.name}")
                    save_tiff(image[top:bottom, left:right], out_base / task / "images" / frame["framed_file"])
                    mask_path = src_masks / f"{source.stem}.png"
                    mask = np.asarray(Image.open(mask_path), dtype=np.uint8)
                    if mask.shape != image.shape:
                        raise ValueError(f"Mask/image size mismatch before framing: {source.name}")
                    (out_base / task / "masks").mkdir(parents=True, exist_ok=True)
                    cropped_mask_path = out_base / task / "masks" / f"{Path(frame['framed_file']).stem}.png"
                    Image.fromarray(mask[top:bottom, left:right]).save(cropped_mask_path)
                    frame["task_mask_sha256"][task] = sha256_file(cropped_mask_path)

            for task in ("vertebra", "femoral"):
                val_path = base / task / "val_names.txt"
                if val_path.is_file():
                    names = [line.strip() for line in val_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    mapped = [frame_by_original_file[name]["framed_file"] for name in names]
                    validation_names[task].extend(mapped)

            (out_base / "geometry").mkdir(parents=True, exist_ok=True)
            (out_base / "geometry" / f"frame_manifest_{split}.json").write_text(json.dumps({"studies": frame_rows}, indent=2) + "\n", encoding="utf-8")

        for task, names in validation_names.items():
            if names:
                (args.out / task).mkdir(parents=True, exist_ok=True)
                (args.out / task / "val_names.txt").write_text("\n".join(sorted(set(names))) + "\n", encoding="utf-8")

        manifest = {
            "schema_version": "spinecontour-spinopelvic.frame-manifest/1",
            "source_dataset": str(args.dataset),
            "framing_implementation": "backend.framing.reframe/fallback_window",
            "model_frame_tasks": ["vertebra", "hrnet", "femoral"],
            "s1_task": "unframed original-image training data; S1 is used to localize candidate crops at inference",
            "studies": all_frames,
            "note": "Contains coded study IDs and crop windows. Keep in approved secure storage. Framed model predictions must be translated back by adding window left/top before geometry analysis.",
        }
        (args.out / "frame_manifest_all.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"framed_studies": len(all_frames), "output": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
