#!/usr/bin/env python3
"""Generate empty research annotation records from a coded CSV manifest.

The CSV and generated records must remain in approved secure storage. Image SHA-256
is calculated over the exact source file bytes, matching Spine-Contour's bundle-io
hashFiles/sha256File implementation (not a re-encoded or pixel-array hash).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np

LEVELS = ("L1", "L2", "L3", "L4", "L5")
CORNERS = ("SA", "SP", "IA", "IP")
LANDMARKS = [f"{level}_{corner}" for level in LEVELS for corner in CORNERS] + ["S1_SA", "S1_SP", "FH1_CENTER", "FH2_CENTER"]
MASK_NAMES = (*LEVELS, "FEMORAL_REGION_UNION")
STAGES = {"pilot", "development", "locked_test_reference"}
SIDES = {"image_left", "image_right", "unknown"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not result:
        raise ValueError("study_id and reader_code must produce a safe filename")
    return result[:100]


def image_info(path: Path) -> tuple[int, int, list[float] | None]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size[0], image.size[1], None
    except Exception:
        # Match the parent application backend.server._dicom_pixel_array:
        # one grayscale frame, RescaleSlope/Intercept, MONOCHROME1 inversion.
        try:
            import pydicom
            dataset = pydicom.dcmread(str(path))
            pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
            if pixels.ndim != 2:
                raise ValueError("Only single-frame grayscale DICOM files are supported")
            pixels = pixels * float(getattr(dataset, "RescaleSlope", 1.0)) + float(getattr(dataset, "RescaleIntercept", 0.0))
            if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
                pixels = float(pixels.max() + pixels.min()) - pixels
            spacing = getattr(dataset, "PixelSpacing", None)
            spacing_xy = [float(spacing[1]), float(spacing[0])] if spacing is not None and len(spacing) == 2 else None
            return int(pixels.shape[1]), int(pixels.shape[0]), spacing_xy
        except Exception as exc:
            raise ValueError(f"Cannot read raster image or source-compatible grayscale DICOM {path.name}: {exc}") from exc


def make_record(row: dict[str, str], image_path: Path, protocol_version: str) -> dict[str, Any]:
    width, height, dicom_spacing = image_info(image_path)
    side = (row.get("anterior_side") or "unknown").strip() or "unknown"
    if side not in SIDES:
        raise ValueError(f"Invalid anterior_side {side!r}; expected one of {sorted(SIDES)}")
    stage = (row.get("stage") or "").strip()
    if stage not in STAGES:
        raise ValueError(f"Invalid stage {stage!r}; expected one of {sorted(STAGES)}")
    spacing = dicom_spacing
    sx, sy = (row.get("pixel_spacing_x_mm") or "").strip(), (row.get("pixel_spacing_y_mm") or "").strip()
    if sx or sy:
        if not sx or not sy:
            raise ValueError("Both pixel_spacing_x_mm and pixel_spacing_y_mm must be supplied together")
        spacing = [float(sx), float(sy)]
        if not all(math.isfinite(v) and v > 0 for v in spacing):
            raise ValueError("Pixel spacing must be finite and positive")
    return {
        "schema_version": "spinecontour-spinopelvic.annotation/0.2",
        "study_id": (row.get("study_id") or "").strip(),
        "image": {
            "sha256": sha256_file(image_path),
            "width_px": width,
            "height_px": height,
            "view": "standing_lateral",
            "anterior_side": side,
            "pixel_spacing_mm": spacing,
        },
        "annotation": {
            "reader_code": (row.get("reader_code") or "").strip(),
            "protocol_version": protocol_version,
            "stage": stage,
            "patient_group_code": (row.get("patient_group_code") or "").strip(),
        },
        "model": None,
        "masks": {name: {"status": "not_placed", "polygons": []} for name in MASK_NAMES},
        "s1_detection": {"status": "not_placed", "bbox": None, "uncertainty_note": None},
        "landmarks": {
            name: {"status": "not_placed", "point": None, "uncertainty": "not_assessable", "uncertainty_note": None}
            for name in LANDMARKS
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Secure local folder of de-identified images")
    parser.add_argument("--metadata-csv", type=Path, required=True, help="Coded CSV: study_id,patient_group_code,reader_code,stage,image_file,anterior_side; PNG/JPEG/TIFF or single-frame grayscale DICOM")
    parser.add_argument("--out", type=Path, required=True, help="Secure local output folder outside the Git repository")
    parser.add_argument("--schema", type=Path, default=Path(__file__).resolve().parents[1] / "schemas" / "annotation.schema.json")
    parser.add_argument("--protocol-version", default="draft-0.2")
    parser.add_argument("--allow-in-repo", action="store_true", help="Only for synthetic tests; never use with real data")
    args = parser.parse_args()
    try:
        if not args.allow_in_repo:
            root = Path(__file__).resolve().parents[3].resolve()
            out = args.out.expanduser().resolve()
            try:
                out.relative_to(root)
            except ValueError:
                pass
            else:
                raise ValueError(f"Refusing to write annotation templates inside the Git repository: {out}")
        try:
            from jsonschema import Draft202012Validator
        except ImportError as exc:
            raise RuntimeError("Install jsonschema: pip install -r research/spinopelvic/requirements-research.txt") from exc
        schema = json.loads(args.schema.read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
        required = {"study_id", "patient_group_code", "reader_code", "stage", "image_file", "anterior_side"}
        args.out.mkdir(parents=True, exist_ok=True)
        created, seen = [], set()
        with args.metadata_csv.open(newline="", encoding="utf-8-sig") as handle:
            rows = csv.DictReader(handle)
            if not rows.fieldnames or not required.issubset(set(rows.fieldnames)):
                raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
            for row in rows:
                sid, reader = (row.get("study_id") or "").strip(), (row.get("reader_code") or "").strip()
                patient = (row.get("patient_group_code") or "").strip()
                if not sid or not patient or not reader:
                    raise ValueError("study_id, patient_group_code and reader_code must be non-empty coded values")
                pair = (sid, reader)
                if pair in seen:
                    raise ValueError(f"Duplicate study_id/reader_code row: {sid} / {reader}")
                seen.add(pair)
                rel = Path((row.get("image_file") or "").strip())
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError(f"image_file must be a relative path inside --images: {rel}")
                image_path = (args.images / rel).resolve()
                try:
                    image_path.relative_to(args.images.resolve())
                except ValueError:
                    raise ValueError(f"Image path escapes --images: {rel}")
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                record = make_record(row, image_path, args.protocol_version)
                errors = list(validator.iter_errors(record))
                if errors:
                    raise ValueError(f"Generated record for {sid} failed schema validation: {errors[0].message}")
                name = f"{safe_name(sid)}__{safe_name(reader)}.annotation.json"
                destination = args.out / name
                if destination.exists():
                    raise FileExistsError(destination)
                destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                created.append(name)
        if not created:
            raise ValueError("No metadata rows found")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"created": len(created), "output_dir": str(args.out), "files": created}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
