#!/usr/bin/env python3
"""Run synthetic round-trip checks on the parent app's S1 crop/coordinate transforms.

No radiographs or model checkpoints are loaded. Uses backend.framing.prepare_crop,
CropTransform.restore_points/restore_mask, reframe, accept_reframe and search_windows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from backend.framing import prepare_crop, reframe, accept_reframe, search_windows  # noqa: E402


def main() -> int:
    height, width = 1600, 1200
    yy, xx = np.mgrid[0:height, 0:width]
    image = ((xx * 0.3 + yy * 0.7) % 4096).astype(np.uint16)
    window = (100, 100, 700, 1500)
    canvas, transform = prepare_crop(image, window)
    if canvas.shape != (768, 768):
        raise AssertionError(f"Unexpected model canvas: {canvas.shape}")

    source_points = np.asarray([[200.0, 300.0], [650.0, 1400.0]], dtype=np.float64)
    left, top, _, _ = window
    model_points = source_points.copy()
    model_points[:, 0] = (model_points[:, 0] - left) * transform.inner.scale + transform.inner.left
    model_points[:, 1] = (model_points[:, 1] - top) * transform.inner.scale + transform.inner.top
    restored_points = transform.restore_points(model_points)
    point_error = float(np.max(np.abs(restored_points - source_points)))
    if point_error > 1e-6:
        raise AssertionError(f"Point inverse transform error too large: {point_error}")

    source_mask = np.zeros((height, width), dtype=np.uint8)
    source_mask[400:700, 200:500] = 1
    crop = source_mask[window[1]:window[3], window[0]:window[2]]
    resized = cv2.resize(crop, (transform.inner.resized_width, transform.inner.resized_height), interpolation=cv2.INTER_NEAREST)
    model_mask = np.zeros((768, 768), dtype=np.uint8)
    y0, x0 = transform.inner.top, transform.inner.left
    model_mask[y0:y0 + transform.inner.resized_height, x0:x0 + transform.inner.resized_width] = resized
    restored_mask = transform.restore_mask(model_mask, (height, width))
    outside = restored_mask.copy()
    outside[window[1]:window[3], window[0]:window[2]] = 0
    if outside.any():
        raise AssertionError("Restored crop mask leaked outside the source crop window")
    intersection = np.logical_and(restored_mask > 0, source_mask > 0).sum()
    union = np.logical_or(restored_mask > 0, source_mask > 0).sum()
    mask_iou = float(intersection / union) if union else 1.0
    if mask_iou < 0.90:
        raise AssertionError(f"Mask round-trip IoU below tolerance: {mask_iou}")

    s1 = np.asarray([[500.0, 1000.0], [560.0, 1000.0]], dtype=np.float64)
    proposed = reframe(s1, (height, width))
    if proposed is None:
        raise AssertionError("Source framing rejected a valid synthetic S1 endplate")
    if not accept_reframe(proposed, proposed):
        raise AssertionError("Identical S1 reframe was not accepted")
    left2, top2, right2, bottom2 = proposed
    shifted = (left2, top2 + 500, right2, bottom2 + 500)
    if accept_reframe(proposed, shifted):
        raise AssertionError("Large S1 crop displacement should have been rejected")

    windows = search_windows(height, width)
    if not windows:
        raise AssertionError("S1 search produced no synthetic candidate windows")
    for candidate in windows:
        l, t, r, b = candidate
        if not (0 <= l < r <= width and 0 <= t < b <= height):
            raise AssertionError(f"Search window outside source image: {candidate}")

    report = {
        "synthetic_only": True,
        "canvas_shape": list(canvas.shape),
        "point_round_trip_max_abs_error_px": point_error,
        "mask_round_trip_iou": mask_iou,
        "s1_reframe_window": list(proposed),
        "search_window_count": len(windows),
        "large_shift_rejected": True,
        "result": "PASS",
        "note": "This checks transform math only; it does not validate S1 detector predictions or clinical framing accuracy.",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
