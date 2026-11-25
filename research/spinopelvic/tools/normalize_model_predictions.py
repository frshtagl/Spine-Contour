#!/usr/bin/env python3
"""Normalize recovered model outputs into the research annotation schema.

Accepts the recovered HRNet/S1 predictions.json formats and original-frame U-Net
label masks. Raw predictions are stored separately from human-reviewed points/boxes.
The output contains case-level coordinates and must stay in approved secure storage.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from build_training_inputs import index_images, safe_stem, sha256_file, image_dimensions  # noqa: E402
from validate_annotations import validate_record  # noqa: E402

LEVELS = ("L1", "L2", "L3", "L4", "L5")
HRNET_NAMES = [f"{level}_{corner}" for level in LEVELS for corner in ("SA", "SP", "IA", "IP")] + ["S1_SA", "S1_SP"]


def load_prediction_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("images")
    if not isinstance(items, list):
        raise ValueError(f"Expected an 'images' list in {path.name}")
    result = {}
    for item in items:
        if not isinstance(item, dict) or not item.get("file"):
            raise ValueError(f"Malformed prediction item in {path.name}")
        name = Path(str(item["file"])).name
        if name in result:
            raise ValueError(f"Duplicate prediction entry for {name} in {path.name}")
        result[name] = item
    return result


def load_frame_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "spinecontour-spinopelvic.frame-manifest/1" or not isinstance(data.get("studies"), list):
        raise ValueError("Invalid source-frame manifest")
    frames = {}
    for item in data["studies"]:
        sid = str(item.get("study_id") or "")
        if not sid or sid in frames:
            raise ValueError(f"Missing/duplicate study_id in frame manifest: {sid!r}")
        frames[sid] = item
    return frames


def _restore_crop_mask(path: Path, frame: dict[str, Any] | None, original_shape: tuple[int, int]) -> np.ndarray:
    if frame is None:
        return _load_gray_mask(path, original_shape)
    left, top, right, bottom = map(int, frame["window"])
    crop = _load_gray_mask(path, (bottom - top, right - left))
    full = np.zeros(original_shape, dtype=crop.dtype)
    full[top:bottom, left:right] = crop
    return full


def _finite_point(value: Any, width: int, height: int, label: str) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label}: expected [x,y], got {value!r}")
    x, y = float(value[0]), float(value[1])
    if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"{label}: point {value!r} is invalid/outside {width}x{height}")
    return [x, y]


def _finite_score(value: Any, label: str) -> float | None:
    if value is None:
        return None
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{label}: model score must be finite")
    return score


def _base_record(path: Path, model_name: str, model_version: str) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("schema_version") != "spinecontour-spinopelvic.annotation/0.2":
        raise ValueError(f"Not an annotation template: {path.name}")
    record = copy.deepcopy(record)
    record["model"] = {"name": model_name, "version": model_version}
    for mask in record["masks"].values():
        mask["status"] = "not_placed"
        mask["polygons"] = []
        mask["model_polygons"] = None
    for landmark in record["landmarks"].values():
        landmark["status"] = "not_placed"
        landmark["point"] = None
        landmark["model_point"] = None
        landmark["model_score"] = None
        landmark["uncertainty"] = "not_assessable"
        landmark["uncertainty_note"] = None
    record["s1_detection"] = {
        "status": "not_placed", "bbox": None, "model_bbox": None,
        "model_score": None, "uncertainty_note": None,
    }
    return record


def _map_model_filename(study_id: str, source: Path) -> str:
    suffix = ".tiff" if source.suffix.lower() in {".dcm", ".dicom"} else source.suffix.lower()
    return f"{safe_stem(study_id)}{suffix}"


def _find_mask(directory: Path, model_file: str, source: Path) -> Path | None:
    candidates = [directory / f"{Path(model_file).stem}.png", directory / f"{source.stem}.png"]
    return next((path for path in candidates if path.is_file()), None)


def _load_gray_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"Prediction mask must be a 2D integer image: {path.name}")
    if array.shape != expected_shape:
        raise ValueError(f"Prediction mask {path.name} shape {array.shape} != source image {expected_shape}")
    return array


def _apply_hrnet(record: dict[str, Any], item: dict[str, Any] | None, min_raw_score: float, frame: dict[str, Any] | None = None) -> dict[str, Any]:
    if item is None:
        return {"predicted": 0, "accepted": 0, "rejected_below_score": 0, "raw_points": {}}
    width, height = record["image"]["width_px"], record["image"]["height_px"]
    if frame is not None:
        left, top, right, bottom = map(int, frame["window"])
        prediction_width, prediction_height = right - left, bottom - top
    else:
        left = top = 0
        prediction_width, prediction_height = width, height
    points = item.get("landmarks", {})
    scores = item.get("conf", {})
    accepted = rejected = 0
    raw_points = {}
    for name in HRNET_NAMES:
        if name not in points:
            continue
        crop_point = _finite_point(points[name], prediction_width, prediction_height, f"HRNet {name}")
        point = [crop_point[0] + left, crop_point[1] + top] if crop_point is not None else None
        score = _finite_score(scores.get(name), f"HRNet {name}")
        use_point = point is not None and (score is None and min_raw_score <= 0 or score is not None and score >= min_raw_score)
        landmark = record["landmarks"][name]
        landmark["model_point"] = point if use_point else None
        landmark["model_score"] = score
        landmark["uncertainty"] = "uncertain"
        landmark["uncertainty_note"] = "Raw HRNet heatmap peak; score is not calibrated as a probability" if use_point else f"HRNet point withheld by raw-score threshold {min_raw_score}"
        raw_points[name] = {"point": point, "raw_score": score, "accepted_for_geometry": bool(use_point)}
        accepted += int(use_point)
        rejected += int(point is not None and not use_point)
    return {"predicted": len(raw_points), "accepted": accepted, "rejected_below_score": rejected, "raw_points": raw_points}


def _apply_s1(record: dict[str, Any], item: dict[str, Any] | None, min_score: float) -> dict[str, Any]:
    if item is None:
        return {"detected": False, "accepted_for_geometry": False, "raw_keypoints": None}
    width, height = record["image"]["width_px"], record["image"]["height_px"]
    score = _finite_score(item.get("score"), "S1 detector")
    bbox = item.get("bbox")
    if bbox is not None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("S1 bbox must be [x1,y1,x2,y2]")
        bbox = [float(v) for v in bbox]
        if not all(math.isfinite(v) for v in bbox) or bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError(f"S1 bbox outside source image: {bbox}")
        record["s1_detection"]["model_bbox"] = bbox
        record["s1_detection"]["model_score"] = score
    keypoints = item.get("keypoints")
    accepted = score is None and min_score <= 0 or score is not None and score >= min_score
    raw_keypoints = None
    if keypoints is not None:
        if len(keypoints) != 2:
            raise ValueError("S1 prediction must contain exactly two keypoints")
        raw_keypoints = {}
        for name, point in zip(("S1_SA", "S1_SP"), keypoints):
            point = _finite_point(point, width, height, f"S1 {name}")
            raw_keypoints[name] = point
            landmark = record["landmarks"][name]
            landmark["model_point"] = point if accepted else None
            landmark["model_score"] = score
            landmark["uncertainty"] = "uncertain"
            landmark["uncertainty_note"] = "Raw S1 detector keypoint; score is the detection score" if accepted else f"S1 keypoint withheld by score threshold {min_score}"
    return {"detected": bbox is not None and keypoints is not None, "score": score,
            "accepted_for_geometry": bool(accepted), "raw_keypoints": raw_keypoints}


def _apply_vertebra_mask(record: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    if mask.min(initial=0) < 0 or mask.max(initial=0) > 5:
        raise ValueError("Vertebra label mask must contain only integer labels 0..5")
    from backend.utils import vertebral_quadrilaterals
    quadrilaterals = vertebral_quadrilaterals(mask.astype(np.uint8, copy=False))
    anterior = record["image"].get("anterior_side", "unknown")
    for level in LEVELS:
        body = quadrilaterals.get(level)
        mask_record = record["masks"][level]
        quad = [[float(point[0]), float(point[1])] for point in body["quadrilateral"]] if body is not None else None
        mask_record["model_polygons"] = [quad] if quad is not None else []
        if body is None:
            continue
        for edge_name, points in (("superior", body["superior"]), ("inferior", body["inferior"])):
            edge = [list(map(float, point)) for point in points]
            if anterior == "image_right":
                edge = edge[::-1]
            prefix = "S" if edge_name == "superior" else "I"
            for suffix, point in zip(("A", "P"), edge):
                landmark = record["landmarks"][f"{level}_{prefix}{suffix}"]
                landmark["model_point"] = point
                landmark["uncertainty"] = "uncertain"
                landmark["uncertainty_note"] = "Derived from source vertebral_quadrilaterals on the predicted label mask"
    return {"levels_present": sorted(quadrilaterals)}


def _apply_femoral_mask(record: dict[str, Any], mask: np.ndarray, min_fit_confidence: float, min_iou: float) -> dict[str, Any]:
    from backend.femoral import fit_two_discs
    fit = fit_two_discs(mask, polish=True)
    if fit is None:
        return {"fit": None, "accepted": False, "reason": "two_disc_fit_failed"}
    circles = sorted(fit["circles"], key=lambda circle: float(circle[0]))
    confidence = float(fit["fit_confidence"])
    iou = float(fit["iou"])
    accepted = confidence >= min_fit_confidence and iou >= min_iou
    for name, circle in zip(("FH1_CENTER", "FH2_CENTER"), circles):
        landmark = record["landmarks"][name]
        landmark["model_score"] = confidence
        landmark["uncertainty"] = "uncertain"
        landmark["uncertainty_note"] = f"Two-disc fit; method={fit['method']}; IoU={iou:.4f}" if accepted else "Two-disc fit withheld by configured confidence/IoU QC"
        if accepted:
            landmark["model_point"] = [float(circle[0]), float(circle[1])]
    return {"fit": {"method": fit["method"], "cost": float(fit["cost"]),
                     "fit_confidence": confidence, "iou": iou,
                     "circles": [[float(v) for v in circle] for circle in circles]},
            "accepted": accepted,
            "reason": None if accepted else "fit_confidence_or_iou_below_configured_threshold"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-annotations", type=Path, required=True, help="One empty MODEL template per study")
    parser.add_argument("--images", type=Path, required=True, help="Original source images, used for raw-byte SHA-256 matching")
    parser.add_argument("--reader-code", default="MODEL", help="reader_code in the base templates and normalized output")
    parser.add_argument("--hrnet-predictions", type=Path)
    parser.add_argument("--s1-predictions", type=Path)
    parser.add_argument("--vertebra-mask-dir", type=Path)
    parser.add_argument("--femoral-mask-dir", type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--component-manifest", type=Path, help="Optional JSON with component checkpoint hashes")
    parser.add_argument("--frame-manifest", type=Path, help="Optional make_source_frames.py manifest; restores cropped model outputs to original image coordinates")
    parser.add_argument("--hrnet-raw-score-min", type=float, default=0.0, help="Pre-registered minimum raw HRNet heatmap peak; not a probability")
    parser.add_argument("--s1-score-min", type=float, default=0.0, help="Pre-registered minimum S1 detector score")
    parser.add_argument("--min-femoral-fit-confidence", type=float, default=0.0)
    parser.add_argument("--min-femoral-iou", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True, help="Secure output folder outside Git")
    parser.add_argument("--allow-in-repo", action="store_true", help="Only for synthetic tests; never use with real data")
    args = parser.parse_args()
    try:
        if not any((args.hrnet_predictions, args.s1_predictions, args.vertebra_mask_dir, args.femoral_mask_dir)):
            raise ValueError("Supply at least one model prediction input")
        if args.hrnet_predictions and args.vertebra_mask_dir:
            raise ValueError("HRNet and vertebral U-Net both provide L1-L5 landmarks; normalize them as separate model routes to avoid overwriting one prediction with another")
        if not 0 <= args.min_femoral_fit_confidence <= 1 or not 0 <= args.min_femoral_iou <= 1:
            raise ValueError("Femoral fit confidence and IoU thresholds must be in [0,1]")
        if not args.allow_in_repo:
            try:
                args.out.expanduser().resolve().relative_to(ROOT.resolve())
            except ValueError:
                pass
            else:
                raise ValueError("Refusing to write case-level prediction annotations inside Git")
        if args.out.exists() and any(args.out.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {args.out}")
        try:
            from jsonschema import Draft202012Validator
        except ImportError as exc:
            raise RuntimeError("Install research dependencies first") from exc
        schema_path = ROOT / "research" / "spinopelvic" / "schemas" / "annotation.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
        image_index = index_images(args.images)
        hrnet_map = load_prediction_map(args.hrnet_predictions)
        s1_map = load_prediction_map(args.s1_predictions)
        frame_map = load_frame_manifest(args.frame_manifest)
        base_files = sorted(args.base_annotations.rglob("*.json"))
        if not base_files:
            raise ValueError("No base annotation templates found")
        args.out.mkdir(parents=True, exist_ok=True)
        records, study_qc, base_used = [], [], []
        used_ids = set()
        for path in base_files:
            base = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(base, dict) or base.get("schema_version") != "spinecontour-spinopelvic.annotation/0.2":
                continue
            if base.get("annotation", {}).get("reader_code") != args.reader_code:
                continue
            sid = str(base["study_id"])
            if sid in used_ids:
                raise ValueError(f"Duplicate base template for study {sid}")
            used_ids.add(sid)
            digest = base["image"]["sha256"].lower()
            source = image_index.get(digest)
            if source is None:
                raise ValueError(f"No source image matches template hash for study {sid}")
            if image_dimensions(source) != (base["image"]["width_px"], base["image"]["height_px"]):
                raise ValueError(f"Image dimensions disagree for study {sid}")
            record = _base_record(path, args.model_name, args.model_version)
            base_used.append(path)
            model_file = _map_model_filename(sid, source)
            frame = frame_map.get(sid)
            if args.frame_manifest and frame is None:
                raise ValueError(f"Frame manifest has no entry for study {sid}")
            if frame is not None:
                if frame.get("image_sha256", "").lower() != digest:
                    raise ValueError(f"Frame manifest image hash differs for study {sid}")
                if (frame.get("original_width_px"), frame.get("original_height_px")) != (record["image"]["width_px"], record["image"]["height_px"]):
                    raise ValueError(f"Frame manifest dimensions differ for study {sid}")
                left, top, right, bottom = map(int, frame["window"])
                if not (0 <= left < right <= record["image"]["width_px"] and 0 <= top < bottom <= record["image"]["height_px"]):
                    raise ValueError(f"Frame manifest window is outside the source image for study {sid}")
                framed_file = str(frame["framed_file"])
                s1_file = str(frame.get("s1_file") or model_file)
            else:
                framed_file = model_file
                s1_file = model_file
            qc = {"study_id": sid, "image_sha256": digest, "model_file": model_file, "framed_file": framed_file,
                  "frame_window": frame.get("window") if frame else None, "components": {}}
            hrnet_item = hrnet_map.get(framed_file) if frame else (hrnet_map.get(model_file) or hrnet_map.get(source.name))
            s1_item = s1_map.get(s1_file) if frame else (s1_map.get(model_file) or s1_map.get(source.name))
            qc["components"]["hrnet_points"] = _apply_hrnet(record, hrnet_item, args.hrnet_raw_score_min, frame)
            qc["components"]["s1_detected"] = _apply_s1(record, s1_item, args.s1_score_min)
            if args.vertebra_mask_dir:
                mask_path = _find_mask(args.vertebra_mask_dir, framed_file, source)
                if mask_path is not None:
                    mask = _restore_crop_mask(mask_path, frame, (record["image"]["height_px"], record["image"]["width_px"]))
                    qc["components"]["vertebra"] = _apply_vertebra_mask(record, mask)
                    qc["components"]["vertebra"]["mask_sha256"] = sha256_file(mask_path)
                else:
                    qc["components"]["vertebra"] = {"missing": True}
            if args.femoral_mask_dir:
                mask_path = _find_mask(args.femoral_mask_dir, framed_file, source)
                if mask_path is not None:
                    mask = _restore_crop_mask(mask_path, frame, (record["image"]["height_px"], record["image"]["width_px"]))
                    fit_qc = _apply_femoral_mask(record, mask, args.min_femoral_fit_confidence, args.min_femoral_iou)
                    fit_qc["mask_sha256"] = sha256_file(mask_path)
                    qc["components"]["femoral"] = fit_qc
                else:
                    qc["components"]["femoral"] = {"missing": True}
            errors = validate_record(record, validator)
            if errors:
                raise ValueError(f"Normalized prediction record for {sid} failed validation: {errors[0]}")
            filename = f"{safe_stem(sid)}__{safe_stem(args.reader_code)}.prediction.annotation.json"
            (args.out / filename).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            qc["output_annotation"] = filename
            study_qc.append(qc)
            records.append(filename)
        if not records:
            raise ValueError(f"No base templates found for reader_code {args.reader_code!r}")
        component_hashes = json.loads(args.component_manifest.read_text(encoding="utf-8")) if args.component_manifest else None
        manifest = {
            "schema_version": "spinecontour-spinopelvic.prediction-manifest/1",
            "model": {"name": args.model_name, "version": args.model_version},
            "reader_code": args.reader_code,
            "base_annotation_files": len(records),
            "base_annotation_sha256": sorted(sha256_file(p) for p in base_used),
            "source_image_index_count": len(image_index),
            "inputs": {
                "hrnet_predictions_sha256": sha256_file(args.hrnet_predictions) if args.hrnet_predictions else None,
                "s1_predictions_sha256": sha256_file(args.s1_predictions) if args.s1_predictions else None,
                "frame_manifest_sha256": sha256_file(args.frame_manifest) if args.frame_manifest else None,
            },
            "score_thresholds": {"hrnet_raw_score_min": args.hrnet_raw_score_min, "s1_score_min": args.s1_score_min},
            "femoral_fit_thresholds": {"fit_confidence_min": args.min_femoral_fit_confidence, "iou_min": args.min_femoral_iou},
            "precedence": "If both HRNet and S1 predictions are supplied, S1 detector endpoints replace HRNet S1 endpoints; lumbar HRNet and vertebral U-Net routes must be normalized separately.",
            "component_checkpoint_manifest": component_hashes,
            "study_qc": study_qc,
            "output_annotations": records,
            "note": "Contains coded study IDs and predicted coordinates. Keep in approved secure storage. Raw model points are in model_point, separate from human-reviewed point fields.",
        }
        (args.out / "prediction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"normalized_records": len(records), "output": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
