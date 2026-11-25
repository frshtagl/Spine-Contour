#!/usr/bin/env python3
"""Build secure model-specific datasets from the spinopelvic annotation schema.

This builder copies approved local images and rasterizes polygons into model-specific
training formats. Output must be outside Git and remain in approved secure storage.
It deliberately does not copy locked-test data unless --include-test-reference is set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".dicom"}
HRNET_NAMES = [f"{level}_{corner}" for level in ("L1", "L2", "L3", "L4", "L5") for corner in ("SA", "SP", "IA", "IP")] + ["S1_SA", "S1_SP"]
VERTEBRA_LEVELS = ("L1", "L2", "L3", "L4", "L5")
POSITIVE_STATES = {"segmented", "edited", "manual"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not cleaned:
        raise ValueError("study_id cannot be converted to a safe filename")
    return cleaned[:100]


def load_annotation_records(folder: Path, schema_path: Path) -> list[dict[str, Any]]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError("Install research dependencies: pip install -r research/spinopelvic/requirements-research.txt") from exc
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    records = []
    for path in sorted(folder.rglob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict) or obj.get("schema_version") != "spinecontour-spinopelvic.annotation/0.2":
            continue
        errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.absolute_path))
        if errors:
            first = errors[0]
            raise ValueError(f"Schema validation failed in {path.name} at {list(first.absolute_path)}: {first.message}")
        records.append(obj)
    if not records:
        raise ValueError(f"No annotation records found under {folder}")
    return records


def dicom_pixels(path: Path) -> np.ndarray:
    """Decode one grayscale DICOM like backend.server._dicom_pixel_array."""
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError("pydicom is required to build from DICOM files") from exc
    dataset = pydicom.dcmread(str(path))
    pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
    if pixels.ndim != 2:
        raise ValueError(f"Only single-frame grayscale DICOM is supported: {path.name}")
    pixels = pixels * float(getattr(dataset, "RescaleSlope", 1.0)) + float(getattr(dataset, "RescaleIntercept", 0.0))
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        pixels = float(pixels.max() + pixels.min()) - pixels
    return pixels


def image_dimensions(path: Path) -> tuple[int, int]:
    if path.suffix.lower() in {".dcm", ".dicom"}:
        arr = dicom_pixels(path)
        return int(arr.shape[1]), int(arr.shape[0])
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        return im.size


def copy_training_image(source: Path, destination: Path) -> str:
    """Copy raster bytes unchanged; convert DICOM pixels to float TIFF for PIL-based kits."""
    if source.suffix.lower() in {".dcm", ".dicom"}:
        arr = dicom_pixels(source).astype(np.float32)
        destination = destination.with_suffix(".tiff")
        Image.fromarray(arr).save(destination, format="TIFF")
        return destination.name
    shutil.copy2(source, destination)
    return destination.name


def index_images(folder: Path) -> dict[str, Path]:
    by_hash: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = defaultdict(list)
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        digest = sha256_file(path)
        if digest in by_hash:
            duplicates[digest].append(str(path))
        else:
            by_hash[digest] = path
    if duplicates:
        detail = next(iter(duplicates.items()))
        raise ValueError(f"Duplicate image bytes found for SHA-256 {detail[0]}: {by_hash[detail[0]]} and {detail[1][0]}")
    if not by_hash:
        raise ValueError(f"No supported images found under {folder}")
    return by_hash


def _positive(obj: dict[str, Any]) -> bool:
    return obj.get("status") in POSITIVE_STATES


def _rasterize_mask(size: tuple[int, int], polygons: list[list[list[float]]], value: int = 1) -> np.ndarray:
    canvas = Image.new("L", size, 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in polygons:
        if len(polygon) < 3:
            raise ValueError("Polygon needs at least three vertices")
        draw.polygon([tuple(map(float, point)) for point in polygon], fill=value)
    return np.asarray(canvas, dtype=np.uint8)


def vertebra_mask(record: dict[str, Any], size: tuple[int, int]) -> tuple[np.ndarray | None, str | None]:
    masks = record["masks"]
    if any(masks[level]["status"] == "not_placed" for level in VERTEBRA_LEVELS):
        return None, "vertebra_incomplete_not_placed"
    out = np.zeros((size[1], size[0]), dtype=np.uint8)
    for class_id, level in enumerate(VERTEBRA_LEVELS, start=1):
        item = masks[level]
        if item["status"] not in POSITIVE_STATES:
            continue
        binary = _rasterize_mask(size, item["polygons"], 1).astype(bool)
        overlap = binary & (out != 0)
        if overlap.any():
            raise ValueError(f"Overlapping vertebral polygons in study {record['study_id']} ({level})")
        out[binary] = class_id
    return out, None


def femoral_mask(record: dict[str, Any], size: tuple[int, int]) -> tuple[np.ndarray | None, str | None]:
    item = record["masks"]["FEMORAL_REGION_UNION"]
    if item["status"] == "not_placed":
        return None, "femoral_incomplete_not_placed"
    if item["status"] == "not_visible":
        return np.zeros((size[1], size[0]), dtype=np.uint8), None
    return _rasterize_mask(size, item["polygons"], 1), None


def point_for(record: dict[str, Any], name: str) -> list[float] | None:
    item = record["landmarks"][name]
    return item["point"] if _positive(item) else None


def make_model_item(record: dict[str, Any], file_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    landmarks = {name: point_for(record, name) for name in HRNET_NAMES}
    hrnet = {"file": file_name, "landmarks": landmarks}
    s1_box = record["s1_detection"]["bbox"] if _positive(record["s1_detection"]) else None
    sa, sp = point_for(record, "S1_SA"), point_for(record, "S1_SP")
    s1 = None
    if s1_box is not None and sa is not None and sp is not None:
        s1 = {"file": file_name, "keypoints": [sa, sp], "bbox": s1_box}
    return hrnet, s1


def input_manifest_sha256(records: list[dict[str, Any]]) -> str:
    """Recompute the study/image/stage fingerprint used when the split was frozen."""
    studies: dict[str, tuple[str, str, str]] = {}
    for record in records:
        annotation = record.get("annotation") or {}
        stage = annotation.get("stage")
        if stage not in {"development", "locked_test_reference"}:
            continue
        sid = str(record.get("study_id") or "")
        digest = str((record.get("image") or {}).get("sha256") or "").lower()
        current = (sid, digest, str(stage))
        prior = studies.setdefault(sid, current)
        if prior != current:
            raise ValueError(f"Reader records disagree on study/image/stage identity for {sid}")
    rows = sorted(studies.values())
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def output_file_manifest(root: Path) -> list[dict[str, Any]]:
    """Hash every generated dataset file; keep this case-level manifest in secure storage."""
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "dataset_build_manifest.json":
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return rows


def _ensure_outside_repo(path: Path, repo_root: Path) -> None:
    resolved, root = path.expanduser().resolve(), repo_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return
    raise ValueError(f"Refusing to write training data inside the Git repository: {resolved}")


def build(records: list[dict[str, Any]], image_index: dict[str, Path], splits: dict[str, Any],
          reader_code: str, output: Path, include_test_reference: bool = False) -> dict[str, Any]:
    split_ids = {"train": set(splits["study_ids"]["train"]), "validation": set(splits["study_ids"]["validation"])}
    if include_test_reference:
        split_ids["test_reference"] = set(splits["study_ids"]["locked_test"])
    if split_ids["train"] & split_ids["validation"]:
        raise ValueError("Split manifest leaks studies between train and validation")
    if include_test_reference and (split_ids["test_reference"] & (split_ids["train"] | split_ids["validation"])):
        raise ValueError("Split manifest leaks locked-test studies into development")
    patient_split: dict[str, str] = {}
    for rec in records:
        sid = rec.get("study_id")
        patient = (rec.get("annotation") or {}).get("patient_group_code")
        if not sid or not patient:
            raise ValueError("Every annotation record must include study_id and patient_group_code")
        for split, ids in split_ids.items():
            if sid in ids:
                expected_stage = "locked_test_reference" if split == "test_reference" else "development"
                if rec["annotation"].get("stage") != expected_stage:
                    raise ValueError(f"Study {sid} is in {split} but its annotation stage is {rec['annotation'].get('stage')!r}")
                prior = patient_split.setdefault(patient, split)
                if prior != split:
                    raise ValueError("Patient-group leakage detected across split manifest")
    selected: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in split_ids}
    for rec in records:
        sid = rec["study_id"]
        for split, ids in split_ids.items():
            if sid in ids and rec["annotation"]["reader_code"] == reader_code:
                if sid in selected[split]:
                    raise ValueError(f"Duplicate selected reader record for study {sid}")
                selected[split][sid] = rec
    for split, ids in split_ids.items():
        missing = ids - set(selected[split])
        if missing:
            raise ValueError(f"Missing reader {reader_code!r} annotation records for {len(missing)} {split} studies")

    counts: dict[str, Any] = {"reader_code": reader_code, "splits": {}, "skipped": {}, "image_sha256_indexed": len(image_index)}
    used_names: set[str] = set()
    for split, by_study in selected.items():
        base = output / "test_reference" if split == "test_reference" else output
        dirs = {
            "vertebra_images": base / "vertebra" / "images",
            "vertebra_masks": base / "vertebra" / "masks",
            "femoral_images": base / "femoral" / "images",
            "femoral_masks": base / "femoral" / "masks",
            "hrnet_images": base / "hrnet" / "images",
            "s1_images": base / "s1" / "images",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        hrnet_items, s1_items, reference_geometry_items = [], [], []
        val_vertebra_names, val_femoral_names = [], []
        task_counts = Counter()
        skipped = Counter()
        for sid, rec in sorted(by_study.items()):
            digest = rec["image"]["sha256"].lower()
            source = image_index.get(digest)
            if source is None:
                raise ValueError(f"No local image matches SHA-256 for study {sid}")
            source_size = image_dimensions(source)
            if source_size != (rec["image"]["width_px"], rec["image"]["height_px"]):
                raise ValueError(f"Image dimensions disagree with annotation for study {sid}: {source_size}")
            size = source_size
            output_suffix = ".tiff" if source.suffix.lower() in {".dcm", ".dicom"} else source.suffix.lower()
            file_name = f"{safe_stem(sid)}{output_suffix}"
            if file_name.casefold() in used_names:
                raise ValueError(f"study_id filename collision after sanitization: {sid}")
            used_names.add(file_name.casefold())
            hrnet_dst = dirs["hrnet_images"] / file_name
            s1_dst = dirs["s1_images"] / file_name
            if not hrnet_dst.exists():
                copy_training_image(source, hrnet_dst)
            if not s1_dst.exists():
                copy_training_image(source, s1_dst)
            hrnet_item, s1_item = make_model_item(rec, file_name)
            fh1, fh2 = point_for(rec, "FH1_CENTER"), point_for(rec, "FH2_CENTER")
            reference_geometry_items.append({
                "study_id": sid,
                "patient_group_code": rec["annotation"]["patient_group_code"],
                "reader_code": rec["annotation"]["reader_code"],
                "stage": rec["annotation"]["stage"],
                "file": file_name,
                "image_sha256": digest,
                "femoral_centres": {"FH1_CENTER": fh1, "FH2_CENTER": fh2},
                "femoral_centres_complete": fh1 is not None and fh2 is not None,
            })
            if fh1 is not None and fh2 is not None:
                task_counts["femoral_reference_centres"] += 1
            if any(value is not None for value in hrnet_item["landmarks"].values()):
                hrnet_items.append(hrnet_item)
                task_counts["hrnet"] += 1
            else:
                skipped["hrnet_no_placed_landmarks"] += 1
            if s1_item is not None:
                s1_items.append(s1_item)
                task_counts["s1"] += 1
            else:
                skipped["s1_missing_box_or_keypoints"] += 1

            vmask, vskip = vertebra_mask(rec, size)
            if vmask is not None:
                dst = dirs["vertebra_images"] / file_name
                if not dst.exists():
                    copy_training_image(source, dst)
                Image.fromarray(vmask).save(dirs["vertebra_masks"] / f"{Path(file_name).stem}.png")
                task_counts["vertebra"] += 1
                if split == "validation":
                    val_vertebra_names.append(file_name)
            else:
                skipped[vskip or "vertebra_skipped"] += 1

            fmask, fskip = femoral_mask(rec, size)
            if fmask is not None:
                dst = dirs["femoral_images"] / file_name
                if not dst.exists():
                    copy_training_image(source, dst)
                Image.fromarray(fmask).save(dirs["femoral_masks"] / f"{Path(file_name).stem}.png")
                task_counts["femoral"] += 1
                if split == "validation":
                    val_femoral_names.append(file_name)
            else:
                skipped[fskip or "femoral_skipped"] += 1

        _write_json(base / "hrnet" / f"annotations_{split}.json", {"images": hrnet_items})
        _write_json(base / "s1" / f"annotations_{split}.json", {"images": s1_items})
        _write_json(base / "geometry" / f"reference_geometry_{split}.json", {"studies": reference_geometry_items})
        if split == "validation":
            (output / "vertebra" / "val_names.txt").write_text("\n".join(sorted(val_vertebra_names)) + "\n", encoding="utf-8")
            (output / "femoral" / "val_names.txt").write_text("\n".join(sorted(val_femoral_names)) + "\n", encoding="utf-8")
        counts["splits"][split] = {"studies": len(by_study), "task_examples": dict(task_counts)}
        counts["skipped"][split] = dict(skipped)

    counts["notes"] = [
        "Images and annotations were copied/rasterized to the requested local output folder; keep it outside Git.",
        "Raster images are copied byte-for-byte. Single-frame grayscale DICOM is decoded with the parent app's slope/intercept and MONOCHROME1 inversion, then saved as float TIFF for the PIL-based training kits; the annotation hash still verifies the original DICOM bytes.",
        "U-Net examples with any L1-L5 status not_placed are excluded rather than mislabeled as background.",
        "Explicit not_visible masks are treated as negative/background; not_placed is never treated as negative.",
        "HRNet uses the source kit's 22 landmark channels. FH1/FH2 centres are retained separately in geometry/reference_geometry_<split>.json and are not HRNet channels.",
        "S1 Keypoint R-CNN examples require a positive S1 box and both S1 endpoints.",
        "The recovered S1 kit derives its training box from keypoints plus its configured padding; the provided annotated bbox is retained in JSON for audit but its current KpDataset does not consume it.",
        "The locked test set is excluded unless --include-test-reference is explicitly supplied; test_reference outputs are written under a physically separate folder and must never be passed to a training command.",
    ]
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Secure local folder of de-identified source images")
    parser.add_argument("--annotations", type=Path, required=True, help="Secure local folder of schema-valid annotation JSON records")
    parser.add_argument("--splits", type=Path, required=True, help="Output from prepare_splits.py")
    parser.add_argument("--reader-code", required=True, help="Use the adjudicated/training reader agreed in the protocol")
    parser.add_argument("--out", type=Path, required=True, help="Secure local output folder outside the Git repository")
    parser.add_argument("--schema", type=Path, default=Path(__file__).resolve().parents[1] / "schemas" / "annotation.schema.json")
    parser.add_argument("--include-test-reference", action="store_true", help="Build a separate test-reference folder; use only after the analysis plan is frozen")
    parser.add_argument("--allow-in-repo", action="store_true", help="Only for synthetic tests; never use with real data")
    args = parser.parse_args()
    try:
        if not args.allow_in_repo:
            _ensure_outside_repo(args.out, Path(__file__).resolve().parents[3])
        records = load_annotation_records(args.annotations, args.schema)
        image_index = index_images(args.images)
        splits = json.loads(args.splits.read_text(encoding="utf-8"))
        split_schema_path = Path(__file__).resolve().parents[1] / "schemas" / "split_manifest.schema.json"
        try:
            from jsonschema import Draft202012Validator
        except ImportError as exc:
            raise RuntimeError("Install jsonschema: pip install -r research/spinopelvic/requirements-research.txt") from exc
        split_schema = json.loads(split_schema_path.read_text(encoding="utf-8"))
        split_errors = list(Draft202012Validator(split_schema).iter_errors(splits))
        if split_errors:
            raise ValueError(f"Invalid split manifest: {split_errors[0].message}")
        current_input_hash = input_manifest_sha256(records)
        if current_input_hash != splits["input_manifest_sha256"]:
            raise ValueError("Annotation study/image/stage manifest differs from the frozen split input hash; regenerate or formally amend the split before building datasets")
        if args.out.exists() and any(args.out.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {args.out}. Choose a new run directory; stale datasets are never merged.")
        args.out.mkdir(parents=True, exist_ok=True)
        report = build(records, image_index, splits, args.reader_code, args.out, args.include_test_reference)
        report["split_manifest_sha256"] = sha256_file(args.splits)
        report["current_input_manifest_sha256"] = current_input_hash
        report["output_files"] = output_file_manifest(args.out)
        report["output_manifest_sha256"] = hashlib.sha256(
            json.dumps(report["output_files"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        _write_json(args.out / "dataset_build_manifest.json", report)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
