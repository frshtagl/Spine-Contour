#!/usr/bin/env python3
"""Summarize model-versus-reference agreement with coverage and clustered bootstrap CIs.

Input CSV columns: study_id, patient_group_code, model_id, parameter,
prediction_deg, reference_deg. withheld_reason is optional. The CSV is a secure
analysis artifact and must not be committed with study-level rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _number(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def icc_a1(pairs: np.ndarray) -> float | None:
    """Two-way random-effects, absolute-agreement, single-measure ICC(A,1)."""
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.shape[0] < 2:
        return None
    n, k = pairs.shape
    grand = float(pairs.mean())
    row_means = pairs.mean(axis=1)
    col_means = pairs.mean(axis=0)
    ms_rows = k * float(((row_means - grand) ** 2).sum()) / (n - 1)
    ms_cols = n * float(((col_means - grand) ** 2).sum()) / (k - 1)
    residual = pairs - row_means[:, None] - col_means[None, :] + grand
    ms_error = float((residual ** 2).sum()) / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if denom == 0:
        return None
    return (ms_rows - ms_error) / denom


def _summary(rows: list[dict[str, Any]], bootstrap: int, seed: int) -> dict[str, Any]:
    reference_rows = [r for r in rows if r["reference"] is not None]
    valid = [r for r in reference_rows if r["prediction"] is not None]
    withheld = [r for r in reference_rows if r["prediction"] is None]
    out: dict[str, Any] = {
        "n_reference": len(reference_rows),
        "n_predicted": len(valid),
        "n_withheld": len(withheld),
        "n_patient_groups": len({r["patient_group_code"] for r in reference_rows}),
        "coverage": len(valid) / len(reference_rows) if reference_rows else None,
        "withheld_reasons": dict(Counter(r["withheld_reason"] or "unspecified" for r in withheld)),
    }
    if not valid:
        out.update({"mae_deg": None, "median_absolute_error_deg": None, "bias_deg": None,
                    "rmse_deg": None, "bland_altman_limits_deg": None, "icc_a1": None,
                    "bootstrap_95ci": None})
        return out
    pred = np.asarray([r["prediction"] for r in valid], dtype=float)
    ref = np.asarray([r["reference"] for r in valid], dtype=float)
    diff = pred - ref
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else None
    out.update({
        "mae_deg": float(np.abs(diff).mean()),
        "median_absolute_error_deg": float(np.median(np.abs(diff))),
        "bias_deg": bias,
        "rmse_deg": float(np.sqrt(np.mean(diff ** 2))),
        "bland_altman_limits_deg": [bias - 1.96 * sd, bias + 1.96 * sd] if sd is not None else None,
        "icc_a1": icc_a1(np.column_stack([pred, ref])),
    })
    if bootstrap <= 0 or len(valid) < 2:
        out["bootstrap_95ci"] = None
        return out
    clusters: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(valid):
        clusters[row["patient_group_code"]].append(idx)
    keys = sorted(clusters)
    rng = random.Random(seed)
    maes, biases, iccs = [], [], []
    for _ in range(bootstrap):
        sampled = [rng.choice(keys) for _ in keys]
        idxs = [i for key in sampled for i in clusters[key]]
        bp, br = pred[idxs], ref[idxs]
        bd = bp - br
        maes.append(float(np.abs(bd).mean()))
        biases.append(float(bd.mean()))
        value = icc_a1(np.column_stack([bp, br]))
        if value is not None and math.isfinite(value):
            iccs.append(float(value))
    def ci(values: list[float]) -> list[float] | None:
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values else None
    out["bootstrap_95ci"] = {"mae_deg": ci(maes), "bias_deg": ci(biases), "icc_a1": ci(iccs),
                             "resamples": bootstrap, "cluster_unit": "patient_group_code"}
    return out


def analyze(rows: list[dict[str, Any]], bootstrap: int = 2000, seed: int = 42) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if not row["model_id"] or not row["parameter"] or not row["study_id"] or not row["patient_group_code"]:
            raise ValueError("study_id, patient_group_code, model_id and parameter must be non-empty")
        for field in ("prediction", "reference"):
            value = row[field]
            if value is not None and not 0.0 <= value <= 180.0:
                raise ValueError(f"{field} for {row['study_id']} / {row['parameter']} is outside 0–180 degrees")
        key = (row["model_id"], row["parameter"], row["study_id"])
        if key in seen:
            raise ValueError(f"Duplicate model/parameter/study row: {key}")
        seen.add(key)
        grouped[(row["model_id"], row["parameter"])].append(row)
    report: dict[str, Any] = {"schema_version": "spinecontour-spinopelvic-agreement/1", "analysis": {}}
    for idx, ((model, parameter), items) in enumerate(sorted(grouped.items())):
        report["analysis"].setdefault(model, {})[parameter] = _summary(items, bootstrap, seed + idx)
    report["pairwise_comparisons"] = pairwise_comparisons(rows, bootstrap, seed)
    report["interpretation_notes"] = [
        "Coverage is reported beside accuracy; withheld predictions are never imputed as zero.",
        "Bland–Altman limits use the SD of paired prediction-minus-reference differences.",
        "ICC(A,1) is a two-way random-effects, absolute-agreement, single-measure ICC; interpret with coverage and error metrics, not alone.",
        "Bootstrap resampling is clustered by coded patient_group_code to account for repeated examinations per patient.",
        "Pairwise model differences use only studies with valid predictions from both models and are bootstrapped by patient group.",
        "This report is descriptive internal validation; it does not establish clinical utility or external validity.",
    ]
    return report


def pairwise_comparisons(rows: list[dict[str, Any]], bootstrap: int, seed: int) -> dict[str, Any]:
    """Paired model comparisons on shared reference studies, clustered by patient."""
    by_parameter: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_parameter[row["parameter"]][row["model_id"]][row["study_id"]] = row
    report: dict[str, Any] = {}
    pair_index = 0
    for parameter, models in sorted(by_parameter.items()):
        model_names = sorted(models)
        if len(model_names) < 2:
            continue
        report[parameter] = {}
        reference_by_study: dict[str, float] = {}
        patient_by_study: dict[str, str] = {}
        for model_map in models.values():
            for sid, row in model_map.items():
                if row["reference"] is None:
                    continue
                if sid in reference_by_study and abs(reference_by_study[sid] - row["reference"]) > 1e-6:
                    raise ValueError(f"Models have different reference values for {sid} / {parameter}")
                if sid in patient_by_study and patient_by_study[sid] != row["patient_group_code"]:
                    raise ValueError(f"Models have different patient-group codes for {sid}")
                reference_by_study[sid] = row["reference"]
                patient_by_study[sid] = row["patient_group_code"]
        for index_a, model_a in enumerate(model_names):
            for model_b in model_names[index_a + 1:]:
                map_a, map_b = models[model_a], models[model_b]
                valid_a = {sid for sid, row in map_a.items() if row["reference"] is not None and row["prediction"] is not None}
                valid_b = {sid for sid, row in map_b.items() if row["reference"] is not None and row["prediction"] is not None}
                matched = sorted(valid_a & valid_b)
                n_reference = len(reference_by_study)
                coverage_a = len(valid_a) / n_reference if n_reference else None
                coverage_b = len(valid_b) / n_reference if n_reference else None
                item: dict[str, Any] = {
                    "n_reference": n_reference,
                    "n_matched_predictions": len(matched),
                    "n_patient_groups_matched": len({patient_by_study[sid] for sid in matched}),
                    "coverage_model_a": coverage_a,
                    "coverage_model_b": coverage_b,
                    "coverage_difference_a_minus_b": coverage_a - coverage_b if coverage_a is not None and coverage_b is not None else None,
                }
                if matched:
                    errors_a = np.asarray([map_a[sid]["prediction"] - reference_by_study[sid] for sid in matched], dtype=float)
                    errors_b = np.asarray([map_b[sid]["prediction"] - reference_by_study[sid] for sid in matched], dtype=float)
                    abs_diff = np.abs(errors_a) - np.abs(errors_b)
                    signed_diff = errors_a - errors_b
                    item["paired_mae_difference_deg_a_minus_b"] = float(abs_diff.mean())
                    item["paired_bias_difference_deg_a_minus_b"] = float(signed_diff.mean())
                else:
                    abs_diff = signed_diff = np.asarray([], dtype=float)
                    item["paired_mae_difference_deg_a_minus_b"] = None
                    item["paired_bias_difference_deg_a_minus_b"] = None
                if bootstrap > 0 and matched:
                    clusters: dict[str, list[int]] = defaultdict(list)
                    for idx, sid in enumerate(matched):
                        clusters[patient_by_study[sid]].append(idx)
                    keys = sorted(clusters)
                    rng = random.Random(seed + pair_index)
                    mae_samples, bias_samples = [], []
                    for _ in range(bootstrap):
                        sampled = [rng.choice(keys) for _ in keys]
                        idxs = [i for key in sampled for i in clusters[key]]
                        mae_samples.append(float(abs_diff[idxs].mean()))
                        bias_samples.append(float(signed_diff[idxs].mean()))
                    item["paired_difference_95ci"] = {
                        "mae_a_minus_b_deg": [float(np.percentile(mae_samples, 2.5)), float(np.percentile(mae_samples, 97.5))],
                        "bias_a_minus_b_deg": [float(np.percentile(bias_samples, 2.5)), float(np.percentile(bias_samples, 97.5))],
                        "resamples": bootstrap,
                        "cluster_unit": "patient_group_code",
                    }
                else:
                    item["paired_difference_95ci"] = None
                report[parameter][f"{model_a}_vs_{model_b}"] = item
                pair_index += 1
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Secure CSV with paired predictions/reference values")
    parser.add_argument("--out", type=Path, required=True, help="Secure JSON report path")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
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
                raise ValueError(f"Refusing to write study-level agreement output inside Git repository: {out}")
        if args.out.exists():
            raise FileExistsError(f"Output already exists: {args.out}; choose a new versioned path")
        required = {"study_id", "patient_group_code", "model_id", "parameter", "prediction_deg", "reference_deg"}
        rows = []
        with args.input.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                raise ValueError(f"Input CSV must contain: {', '.join(sorted(required))}")
            for raw in reader:
                if not raw.get("study_id") or not raw.get("patient_group_code"):
                    raise ValueError("study_id and patient_group_code are required coded grouping fields")
                prediction = _number(raw.get("prediction_deg"))
                reference = _number(raw.get("reference_deg"))
                if str(raw.get("prediction_deg", "")).strip() and prediction is None:
                    raise ValueError(f"Invalid prediction_deg value for study {raw['study_id']}")
                if str(raw.get("reference_deg", "")).strip() and reference is None:
                    raise ValueError(f"Invalid reference_deg value for study {raw['study_id']}")
                rows.append({"study_id": raw["study_id"], "patient_group_code": raw["patient_group_code"],
                             "model_id": raw["model_id"], "parameter": raw["parameter"],
                             "prediction": prediction, "reference": reference,
                             "withheld_reason": raw.get("withheld_reason", "")})
        if not rows:
            raise ValueError("No paired measurement rows found")
        report = analyze(rows, args.bootstrap, args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote agreement report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
