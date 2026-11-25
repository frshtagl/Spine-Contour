#!/usr/bin/env python3
"""Validate research annotation JSON, pixel coordinates and polygon geometry."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "annotation.schema.json"


def _path_text(error: Any) -> str:
    parts = [str(part) for part in error.absolute_path]
    return ".".join(parts) if parts else "<root>"


def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p, eps: float = 1e-9) -> bool:
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
            and abs(_orient(a, b, p)) <= eps)


def _segments_intersect(a, b, c, d, eps: float = 1e-9) -> bool:
    o1, o2, o3, o4 = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True
    return ((abs(o1) <= eps and _on_segment(a, b, c, eps))
            or (abs(o2) <= eps and _on_segment(a, b, d, eps))
            or (abs(o3) <= eps and _on_segment(c, d, a, eps))
            or (abs(o4) <= eps and _on_segment(c, d, b, eps)))


def _polygon_errors(polygon: Any, prefix: str, width: int, height: int) -> list[str]:
    errors: list[str] = []
    if not isinstance(polygon, list) or len(polygon) < 3:
        return [f"{prefix}: polygon must contain at least three vertices"]
    points = []
    for index, point in enumerate(polygon):
        valid = (isinstance(point, list) and len(point) == 2
                 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in point))
        if not valid:
            errors.append(f"{prefix}[{index}]: expected a numeric [x, y] point")
            continue
        p = (float(point[0]), float(point[1]))
        points.append(p)
        if not (0 <= p[0] < width and 0 <= p[1] < height):
            errors.append(f"{prefix}[{index}]: point {point!r} is outside the {width}x{height} image frame")
    if len(points) != len(polygon):
        return errors
    if len(set(points)) < 3:
        errors.append(f"{prefix}: polygon has fewer than three distinct vertices")
    if any(points[i] == points[(i + 1) % len(points)] for i in range(len(points))):
        errors.append(f"{prefix}: polygon contains consecutive duplicate vertices")
    area2 = sum(points[i][0] * points[(i + 1) % len(points)][1]
                - points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points)))
    if abs(area2) <= 1e-6:
        errors.append(f"{prefix}: polygon has zero or near-zero area")
    n = len(points)
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or j == (i + 1) % n or i == (j + 1) % n:
                continue
            c, d = points[j], points[(j + 1) % n]
            if _segments_intersect(a, b, c, d):
                errors.append(f"{prefix}: self-intersection between edges {i} and {j}")
                return errors
    return errors


def validate_record(record: dict[str, Any], validator: Any) -> list[str]:
    """Return schema and image-coordinate/polygon-geometry errors."""
    if not isinstance(record, dict):
        return ["<root>: annotation record must be a JSON object"]
    errors = [f"{_path_text(error)}: {error.message}" for error in validator.iter_errors(record)]
    image = record.get("image", {})
    width, height = image.get("width_px"), image.get("height_px")
    if not isinstance(width, int) or isinstance(width, bool) or not isinstance(height, int) or isinstance(height, bool):
        return errors

    masks = record.get("masks", {})
    if isinstance(masks, dict):
        for mask_name, mask in masks.items():
            if not isinstance(mask, dict):
                continue
            for field in ("polygons", "model_polygons"):
                polygons = mask.get(field)
                if not isinstance(polygons, list):
                    continue
                for polygon_index, polygon in enumerate(polygons):
                    errors.extend(_polygon_errors(polygon, f"masks.{mask_name}.{field}[{polygon_index}]", width, height))

    s1_detection = record.get("s1_detection", {})
    if isinstance(s1_detection, dict):
        score = s1_detection.get("model_score")
        if score is not None and (not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score)):
            errors.append("s1_detection.model_score: expected a finite numeric score")
        for field in ("bbox", "model_bbox"):
            bbox = s1_detection.get(field)
            if bbox is not None:
                if (not isinstance(bbox, list) or len(bbox) != 4
                        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bbox)
                        or bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height
                        or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
                    errors.append(f"s1_detection.{field}: box {bbox!r} is invalid or outside the {width}x{height} image frame")

    landmarks = record.get("landmarks", {})
    if isinstance(landmarks, dict):
        for landmark_name, landmark in landmarks.items():
            if not isinstance(landmark, dict):
                continue
            score = landmark.get("model_score")
            if score is not None and (not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score)):
                errors.append(f"landmarks.{landmark_name}.model_score: expected a finite numeric score")
            for field in ("point", "model_point"):
                point = landmark.get(field)
                if point is None:
                    continue
                valid = (isinstance(point, list) and len(point) == 2
                         and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in point))
                if not valid or not (0 <= point[0] < width and 0 <= point[1] < height):
                    errors.append(f"landmarks.{landmark_name}.{field}: point {point!r} is invalid or outside the {width}x{height} image frame")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path, help="Local folder containing annotation JSON records")
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    args = parser.parse_args()
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        print("Missing dependency. Install research requirements first.", file=sys.stderr)
        return 2
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    files = sorted(args.annotations.rglob("*.json"))
    valid = invalid = 0
    seen_pairs: set[tuple[str, str]] = set()
    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("schema_version") != "spinecontour-spinopelvic.annotation/0.2":
                continue
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
    if valid + invalid == 0:
        print(f"No annotation records found in {args.annotations}", file=sys.stderr)
        return 2
    print(f"\nSummary: {valid} valid, {invalid} invalid, {valid + invalid} annotation records")
    return 0 if invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
