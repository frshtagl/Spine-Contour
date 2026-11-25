#!/usr/bin/env python3
"""Compute Spine-Contour spinopelvic angles from a research annotation folder.

This calls the fork's existing backend.utils.spinopelvic_measurements_from_geometry
implementation. It is an annotation/reference-geometry utility, not model inference.
The input/output files must remain in approved secure storage.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

LEVELS = ("L1", "L2", "L3", "L4", "L5")


def _point(record: dict[str, Any], name: str, min_model_score: float = 0.0) -> list[float] | None:
    item = record.get("landmarks", {}).get(name, {})
    if item.get("status") in {"segmented", "edited", "manual"}:
        return item.get("point")
    # For a model-output record, keep the raw prediction in model_point and
    # leave the human-reviewed point empty. This allows parallel human/model
    # geometry without conflating the two annotation states.
    if record.get("model") is not None and item.get("model_point") is not None:
        score = item.get("model_score")
        if min_model_score > 0 and (score is None or score < min_model_score):
            return None
        return item["model_point"]
    return None


def geometry_from_record(record: dict[str, Any], min_model_score: float = 0.0) -> tuple[dict[str, Any], list[list[float]] | None, list[list[float]]]:
    vertebrae: dict[str, Any] = {}
    for level in LEVELS:
        sa, sp = _point(record, f"{level}_SA", min_model_score), _point(record, f"{level}_SP", min_model_score)
        ia, ip = _point(record, f"{level}_IA", min_model_score), _point(record, f"{level}_IP", min_model_score)
        superior = [sa, sp] if sa is not None and sp is not None else None
        inferior = [ia, ip] if ia is not None and ip is not None else None
        if superior is None and inferior is None:
            continue
        body: dict[str, Any] = {"superior": superior, "inferior": inferior, "anterior_confirmed": record["image"].get("anterior_side") != "unknown"}
        if superior is not None and inferior is not None:
            body["quadrilateral"] = [sa, sp, ip, ia]
        vertebrae[level] = body
    s1_sa, s1_sp = _point(record, "S1_SA", min_model_score), _point(record, "S1_SP", min_model_score)
    s1 = [s1_sa, s1_sp] if s1_sa is not None and s1_sp is not None else None
    fh1, fh2 = _point(record, "FH1_CENTER", min_model_score), _point(record, "FH2_CENTER", min_model_score)
    # The source geometry API accepts circle triples. The centre coordinates are
    # the only fields used for PI/PT; radius=1 is a neutral shape-valid placeholder,
    # not an estimated anatomical radius and is never reported as a measurement.
    circles = [[fh1[0], fh1[1], 1.0], [fh2[0], fh2[1], 1.0]] if fh1 is not None and fh2 is not None else []
    return vertebrae, s1, circles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--reader-code", required=True)
    parser.add_argument("--min-model-score", type=float, default=0.0, help="Optional pre-registered minimum raw model score; default 0 keeps all finite model points")
    parser.add_argument("--out", type=Path, required=True, help="CSV output in secure storage")
    parser.add_argument("--allow-in-repo", action="store_true", help="Only for synthetic tests; never use with real data")
    args = parser.parse_args()
    try:
        if not math.isfinite(args.min_model_score) or args.min_model_score < 0:
            raise ValueError("--min-model-score must be finite and non-negative")
        if not args.allow_in_repo:
            root = Path(__file__).resolve().parents[3].resolve()
            out = args.out.expanduser().resolve()
            try:
                out.relative_to(root)
            except ValueError:
                pass
            else:
                raise ValueError(f"Refusing to write measurement output inside Git repository: {out}")
        if args.out.exists():
            raise FileExistsError(f"Output already exists: {args.out}; choose a new versioned path")
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from backend.utils import spinopelvic_measurements_from_geometry
        rows = []
        for path in sorted(args.annotations.rglob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("schema_version") != "spinecontour-spinopelvic.annotation/0.2":
                continue
            if record.get("annotation", {}).get("reader_code") != args.reader_code:
                continue
            vertebrae, s1, circles = geometry_from_record(record, args.min_model_score)
            try:
                result = spinopelvic_measurements_from_geometry(vertebrae, s1, circles, allow_empty=True)
                values = result["measurements"]
                rows.append({
                    "study_id": record["study_id"],
                    "patient_group_code": record["annotation"].get("patient_group_code"),
                    "image_sha256": record["image"]["sha256"],
                    "reader_code": args.reader_code,
                    "stage": record["annotation"]["stage"],
                    "min_model_score": args.min_model_score,
                    "PI_deg": values["PI"],
                    "PT_deg": values["PT"],
                    "SS_deg": values["SS"],
                    "LL_L1_S1_deg": values["LL"].get("L1-S1"),
                    "LL_L2_S1_deg": values["LL"].get("L2-S1"),
                    "LL_L3_S1_deg": values["LL"].get("L3-S1"),
                    "LL_L4_S1_deg": values["LL"].get("L4-S1"),
                    "LL_L5_S1_deg": values["LL"].get("L5-S1"),
                    "coverage_partial": result["qc"]["coverage"]["partial"],
                    "coverage_missing": ";".join(result["qc"]["coverage"]["missing"]),
                    "femoral_centres_available": len(circles) == 2,
                    "geometry_error": "",
                })
            except Exception as exc:
                rows.append({"study_id": record["study_id"], "patient_group_code": record["annotation"].get("patient_group_code"),
                             "image_sha256": record["image"]["sha256"], "reader_code": args.reader_code,
                             "stage": record["annotation"]["stage"], "geometry_error": f"{type(exc).__name__}: {exc}"})
        if not rows:
            raise ValueError(f"No annotation records found for reader_code {args.reader_code!r}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({key for row in rows for key in row})
        with args.out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {len(rows)} geometry-derived measurement rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
