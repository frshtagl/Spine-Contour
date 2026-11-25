#!/usr/bin/env python3
"""Create a deterministic patient-disjoint 240/60/100 split from annotation records.

The input annotations and output split manifest must live in approved secure storage,
not in this public Git repository. One JSON annotation record per reader is supported;
all records for a study must agree on patient_group_code, stage and image SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_records(folder: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(folder.rglob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Cannot parse annotation JSON {path.name}: {exc}") from exc
        if not isinstance(obj, dict) or "study_id" not in obj or "image" not in obj or "annotation" not in obj:
            if isinstance(obj, dict) and "study_id" in obj:
                raise ValueError(f"Malformed annotation record: {path.name}")
            continue
        records.append(obj)
    if not records:
        raise ValueError(f"No annotation records found under {folder}")
    return records


def _study_index(records: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    studies: dict[str, dict[str, str]] = {}
    readers: dict[str, set[str]] = defaultdict(set)
    image_owner: dict[str, str] = {}
    for rec in records:
        sid = str(rec.get("study_id", "")).strip()
        ann = rec.get("annotation") or {}
        image = rec.get("image") or {}
        patient = str(ann.get("patient_group_code") or "").strip()
        stage = str(ann.get("stage") or "").strip()
        digest = str(image.get("sha256") or "").strip().lower()
        reader = str(ann.get("reader_code") or "").strip()
        if not sid or not patient:
            raise ValueError("Every record needs non-empty study_id and annotation.patient_group_code for leakage-safe splitting.")
        if stage not in {"pilot", "development", "locked_test_reference"}:
            raise ValueError(f"Study {sid}: unsupported stage {stage!r}")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Study {sid}: image.sha256 must be a 64-character SHA-256 hex digest")
        if not reader:
            raise ValueError(f"Study {sid}: annotation.reader_code is required")
        current = {"patient_group_code": patient, "stage": stage, "image_sha256": digest}
        prior_owner = image_owner.setdefault(digest, sid)
        if prior_owner != sid:
            raise ValueError(f"The same source image hash is assigned to multiple study IDs: {prior_owner} and {sid}")
        if sid in studies and studies[sid] != current:
            raise ValueError(f"Reader records disagree on patient/stage/image identity for study {sid}")
        studies[sid] = current
        if reader in readers[sid]:
            raise ValueError(f"Duplicate reader_code {reader!r} for study {sid}")
        readers[sid].add(reader)
    return studies


def _subset_sum(groups: list[tuple[str, list[str]]], target: int, seed: int) -> set[str]:
    """Select whole patient groups whose study counts sum exactly to target."""
    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    previous: dict[int, tuple[int, str] | None] = {0: None}
    for patient_code, study_ids in shuffled:
        size = len(study_ids)
        for total in sorted(list(previous.keys()), reverse=True):
            new_total = total + size
            if new_total <= target and new_total not in previous:
                previous[new_total] = (total, patient_code)
        if target in previous:
            break
    if target not in previous:
        sizes = sorted((len(ids) for _, ids in groups), reverse=True)
        raise ValueError(
            f"Cannot form an exact {target}-study validation split without splitting patients. "
            f"Patient-group sizes (largest first): {sizes}. Adjust the cohort allocation or collect a compatible cohort."
        )
    selected: set[str] = set()
    total = target
    while total:
        step = previous[total]
        if step is None:
            raise RuntimeError("Split subset reconstruction failed")
        prior_total, patient_code = step
        selected.add(patient_code)
        total = prior_total
    return selected


def _ensure_outside_repo(path: Path, repo_root: Path, allow: bool) -> None:
    if allow:
        return
    resolved = path.expanduser().resolve()
    root = repo_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return
    raise ValueError(f"Refusing to write split manifest inside the Git repository ({resolved}). Use approved secure storage outside the repo.")


def make_split(records: list[dict[str, Any]], train_n: int, val_n: int, test_n: int, seed: int) -> dict[str, Any]:
    studies = _study_index(records)
    development = {sid: x for sid, x in studies.items() if x["stage"] == "development"}
    locked_test = {sid: x for sid, x in studies.items() if x["stage"] == "locked_test_reference"}
    if len(development) != train_n + val_n:
        raise ValueError(f"Expected {train_n + val_n} unique development studies; found {len(development)}")
    if len(locked_test) != test_n:
        raise ValueError(f"Expected {test_n} unique locked-test studies; found {len(locked_test)}")
    dev_groups: dict[str, list[str]] = defaultdict(list)
    test_groups: dict[str, list[str]] = defaultdict(list)
    for sid, row in development.items():
        dev_groups[row["patient_group_code"]].append(sid)
    for sid, row in locked_test.items():
        test_groups[row["patient_group_code"]].append(sid)
    overlap = set(dev_groups) & set(test_groups)
    if overlap:
        raise ValueError(f"Patient leakage: {len(overlap)} coded patient groups occur in both development and locked test")
    val_patients = _subset_sum(list(dev_groups.items()), val_n, seed)
    train_ids, val_ids = [], []
    for patient, ids in dev_groups.items():
        (val_ids if patient in val_patients else train_ids).extend(ids)
    test_ids = [sid for ids in test_groups.values() for sid in ids]
    train_ids.sort(); val_ids.sort(); test_ids.sort()
    if len(train_ids) != train_n or len(val_ids) != val_n or len(test_ids) != test_n:
        raise RuntimeError("Generated split counts do not match requested counts")
    assignment = {}
    for patient in dev_groups:
        assignment[patient] = "validation" if patient in val_patients else "train"
    for patient in test_groups:
        assignment[patient] = "locked_test"
    assignment_sha = hashlib.sha256(json.dumps(assignment, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    input_rows = sorted((sid, row["image_sha256"], row["stage"]) for sid, row in studies.items() if row["stage"] in {"development", "locked_test_reference"})
    input_sha = hashlib.sha256(json.dumps(input_rows, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema_version": "spinecontour-spinopelvic.splits/1",
        "seed": seed,
        "counts": {"train": train_n, "validation": val_n, "locked_test": test_n, "total": train_n + val_n + test_n},
        "patient_group_counts": {
            "train": sum(1 for value in assignment.values() if value == "train"),
            "validation": sum(1 for value in assignment.values() if value == "validation"),
            "locked_test": sum(1 for value in assignment.values() if value == "locked_test"),
        },
        "study_ids": {"train": train_ids, "validation": val_ids, "locked_test": test_ids},
        "assignment_sha256": assignment_sha,
        "input_manifest_sha256": input_sha,
        "locked_test_access": "reference_only; do not use for training, model selection, thresholding or iterative debugging",
        "note": "Patient-group codes are not stored in this manifest. The manifest itself still contains coded study IDs and must remain in approved secure storage.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="Secure local folder containing annotation JSON records")
    parser.add_argument("--out", type=Path, required=True, help="Secure local output JSON path, outside the repository")
    parser.add_argument("--train", type=int, default=240)
    parser.add_argument("--validation", type=int, default=60)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True, help="Freeze and record this in the approved protocol")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing split manifest after protocol review")
    parser.add_argument("--allow-in-repo", action="store_true", help="Only for synthetic tests; never use with real data")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    try:
        _ensure_outside_repo(args.out, repo_root, args.allow_in_repo)
        if args.out.exists() and not args.overwrite:
            raise FileExistsError(f"Split manifest already exists: {args.out}. Use --overwrite only after documenting a protocol amendment.")
        records = _load_records(args.annotations)
        result = make_split(records, args.train, args.validation, args.test, args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temp = args.out.with_suffix(args.out.suffix + ".tmp")
        temp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temp.replace(args.out)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.out), "counts": result["counts"], "assignment_sha256": result["assignment_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
