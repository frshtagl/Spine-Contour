#!/usr/bin/env python3
"""Validate draft spinopelvic annotation JSON files without opening patient images.

Run from the repository root:
  python research/spinopelvic/tools/validate_annotations.py /secure/local/annotations

The input directory must remain outside the public Git repository.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "annotation.schema.json"


def _path_text(error: Any) -> str:
    parts = [str(part) for part in error.absolute_path]
    return ".".join(parts) if parts else "<root>"


def validate_record(record: dict[str, Any], validator: Any) -> list[str]:
    """Return schema and image-coordinate errors for one annotation record."""
    errors = [
        f"{_path_text(error)}: {error.message}"
        for error in validator.iter_errors(record)
    ]

    image = record.get("image", {}) if isinstance(record, dict) else {}
    width = image.get("width_px")
    height = image.get("height_px")
    if not isinstance(width, int) or not isinstance(height, int):
        return errors

    def in_bounds(point: Any) -> bool:
        return (
            isinstance(point, list)
            and len(point) == 2
            and all(isinstance(v, (int, float)) for v in point)
            and 0 <= point[0] < width
            and 0 <= point[1] < height
        )

    for mask_name, mask in record.get("masks", {}).items():
        for polygon_index, polygon in enumerate(mask.get("polygons", [])):
            for point_index, point in enumerate(polygon):
                if not in_bounds(point):
                    errors.append(
                        f"masks.{mask_name}.polygons[{polygon_index}][{point_index}]: "
                        f"point {point!r} is outside the {width}x{height} image frame"
                    )

    for landmark_name, landmark in record.get("landmarks", {}).items():
        point = landmark.get("point")
        if point is not None and not in_bounds(point):
            errors.append(
                f"landmarks.{landmark_name}.point: "
                f"point {point!r} is outside the {width}x{height} image frame"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path, help="Local folder containing one annotation JSON per reader/study")
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    args = parser.parse_args()

    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        print("Missing dependency. Install research requirements first.", file=sys.stderr)
        return 2

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    files = sorted(args.annotations.glob("*.json"))
    if not files:
        print(f"No .json annotation files found in {args.annotations}", file=sys.stderr)
        return 2

    valid = 0
    invalid = 0
    seen_pairs: set[tuple[str, str]] = set()
    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            errors = validate_record(record, validator)
            pair = (str(record.get("study_id", "")), str(record.get("annotation", {}).get("reader_code", "")))
            if pair in seen_pairs:
                errors.append(f"duplicate study_id/reader_code pair: {pair[0]} / {pair[1]}")
            seen_pairs.add(pair)
        except Exception as exc:
            errors = [f"could not parse or inspect JSON: {type(exc).__name__}: {exc}"]

        if errors:
            invalid += 1
            print(f"FAIL  {path.name}")
            for error in errors:
                print(f"      - {error}")
        else:
            valid += 1
            print(f"PASS  {path.name}")

    print(f"\nSummary: {valid} valid, {invalid} invalid, {len(files)} total")
    return 0 if invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
