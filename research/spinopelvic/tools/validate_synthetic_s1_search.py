#!/usr/bin/env python3
"""Exercise locate() end-to-end with a synthetic S1 line and deterministic scorer.

No trained detector/checkpoint or patient image is used. The scorer detects the
synthetic bright endplate in each candidate crop and returns model-frame points.
This validates search -> crop -> restore -> fixed-point selection -> full-film window.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from backend.framing import locate, prepare_crop  # noqa: E402


def synthetic_s1_scorer(canvases: list[np.ndarray]):
    outputs = []
    for canvas in canvases:
        ys, xs = np.where(canvas > 180)
        if len(xs) < 8 or int(xs.max()) - int(xs.min()) < 4:
            outputs.append((0.0, None))
            continue
        y = float(np.median(ys))
        outputs.append((0.99, np.asarray([[float(xs.min()), y], [float(xs.max()), y]], dtype=np.float32)))
    return outputs


def main() -> int:
    height, width = 1600, 1200
    image = np.zeros((height, width), dtype=np.uint8)
    image[1148:1152, 580:621] = 255
    result = locate(image, synthetic_s1_scorer)
    if result is None:
        raise AssertionError("Synthetic S1 scorer did not produce a framing candidate")
    left, top, right, bottom = result["window"]
    crop_height, crop_width = bottom - top, right - left
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise AssertionError(f"Returned window is outside source image: {result['window']}")
    if not (500 <= crop_height <= 700):
        raise AssertionError(f"Fixed-point search selected an implausible crop height: {crop_height}")
    if not (left <= 600 <= right and top <= 1150 <= bottom):
        raise AssertionError("Selected crop does not contain the synthetic S1 endplate")
    _, transform = prepare_crop(image, result["window"])
    source_points = np.asarray([[580.0, 1150.0], [620.0, 1150.0]], dtype=np.float64)
    model_points = source_points.copy()
    model_points[:, 0] = (model_points[:, 0] - left) * transform.inner.scale + transform.inner.left
    model_points[:, 1] = (model_points[:, 1] - top) * transform.inner.scale + transform.inner.top
    restored = transform.restore_points(model_points)
    error = float(np.max(np.abs(restored - source_points)))
    if error > 1e-6:
        raise AssertionError(f"Restored S1 endpoints drifted by {error} px")
    report = {
        "synthetic_only": True,
        "window": list(result["window"]),
        "window_width_px": crop_width,
        "window_height_px": crop_height,
        "candidate_count": result["candidates"],
        "search_strong": result["search_strong"],
        "whole_film_won": result["whole_film_won"],
        "s1_length_px": result["s1_length_px"],
        "point_round_trip_max_abs_error_px": error,
        "result": "PASS",
        "note": "Synthetic scorer validates framing geometry only; it does not validate a trained S1 detector or clinical framing accuracy.",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
