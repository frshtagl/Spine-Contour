from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "analysis"))

from prepare_splits import make_split  # noqa: E402
from build_training_inputs import build, sha256_file, input_manifest_sha256  # noqa: E402
from make_annotation_templates import make_record  # noqa: E402
from analyze_agreement import analyze  # noqa: E402
from validate_annotations import validate_record  # noqa: E402


def synthetic_record(study_id: str, patient: str, stage: str, image_path: Path, reader: str = "R1") -> dict:
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    masks = {}
    for name in ("L1", "L2", "L3", "L4", "L5", "FEMORAL_REGION_UNION"):
        if name == "FEMORAL_REGION_UNION":
            poly = [[20, 80], [80, 80], [80, 110], [20, 110]]
        else:
            y = 10 + (int(name[1]) - 1) * 12
            poly = [[10, y], [50, y], [50, y + 8], [10, y + 8]]
        masks[name] = {"status": "manual", "polygons": [poly]}
    landmarks = {}
    for level_i, level in enumerate(("L1", "L2", "L3", "L4", "L5")):
        y = 10 + level_i * 12
        for name, point in {"SA": [10, y], "SP": [50, y], "IA": [10, y + 8], "IP": [50, y + 8]}.items():
            landmarks[f"{level}_{name}"] = {"status": "manual", "point": point, "uncertainty": "clear"}
    landmarks["S1_SA"] = {"status": "manual", "point": [20, 75], "uncertainty": "clear"}
    landmarks["S1_SP"] = {"status": "manual", "point": [70, 75], "uncertainty": "clear"}
    landmarks["FH1_CENTER"] = {"status": "manual", "point": [30, 100], "uncertainty": "clear"}
    landmarks["FH2_CENTER"] = {"status": "manual", "point": [65, 100], "uncertainty": "clear"}
    return {
        "schema_version": "spinecontour-spinopelvic.annotation/0.2",
        "study_id": study_id,
        "image": {"sha256": image_hash, "width_px": 128, "height_px": 128,
                  "view": "standing_lateral", "anterior_side": "image_left", "pixel_spacing_mm": None},
        "annotation": {"reader_code": reader, "protocol_version": "draft-0.2", "stage": stage,
                       "patient_group_code": patient},
        "model": None,
        "masks": masks,
        "s1_detection": {"status": "manual", "bbox": [15, 70, 75, 82], "uncertainty_note": None},
        "landmarks": landmarks,
    }


class ResearchPipelineTests(unittest.TestCase):
    def test_patient_disjoint_exact_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = []
            for idx, (sid, patient, stage) in enumerate([
                ("D1", "P1", "development"), ("D2", "P2", "development"),
                ("D3", "P3", "development"), ("T1", "P4", "locked_test_reference")
            ]):
                image_path = Path(tmp) / f"{sid}.png"
                Image.fromarray(np.full((128, 128), idx, dtype=np.uint8)).save(image_path)
                records.append(synthetic_record(sid, patient, stage, image_path))
            result = make_split(records, train_n=2, val_n=1, test_n=1, seed=17)
            self.assertEqual(result["counts"], {"train": 2, "validation": 1, "locked_test": 1, "total": 4})
            self.assertEqual(len(set(result["study_ids"]["train"]) & set(result["study_ids"]["validation"])), 0)
            self.assertEqual(result["study_ids"]["locked_test"], ["T1"])
            self.assertEqual(result["input_manifest_sha256"], input_manifest_sha256(records))
            split_schema = json.loads((ROOT / "schemas" / "split_manifest.schema.json").read_text(encoding="utf-8"))
            from jsonschema import Draft202012Validator
            self.assertEqual(list(Draft202012Validator(split_schema).iter_errors(result)), [])

    def test_split_input_fingerprint_changes_when_stage_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = []
            for idx, (sid, patient, stage) in enumerate([
                ("D1", "P1", "development"), ("T1", "P2", "locked_test_reference")
            ]):
                image_path = Path(tmp) / f"{sid}.png"
                Image.fromarray(np.full((128, 128), idx, dtype=np.uint8)).save(image_path)
                records.append(synthetic_record(sid, patient, stage, image_path))
            original = input_manifest_sha256(records)
            changed = copy.deepcopy(records)
            changed[0]["annotation"]["stage"] = "locked_test_reference"
            self.assertNotEqual(original, input_manifest_sha256(changed))

    def test_split_input_fingerprint_rejects_inconsistent_reader_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "D1.png"
            Image.fromarray(np.zeros((128, 128), dtype=np.uint8)).save(image_path)
            first = synthetic_record("D1", "P1", "development", image_path, "R1")
            second = copy.deepcopy(first)
            second["annotation"]["reader_code"] = "R2"
            second["image"]["sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "disagree on study/image/stage identity"):
                input_manifest_sha256([first, second])

    def test_split_refuses_patient_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = []
            for idx, (sid, patient, stage) in enumerate([
                ("D1", "SAME", "development"), ("D2", "P2", "development"),
                ("D3", "P3", "development"), ("T1", "SAME", "locked_test_reference")
            ]):
                image_path = Path(tmp) / f"{sid}.png"
                Image.fromarray(np.full((128, 128), idx, dtype=np.uint8)).save(image_path)
                records.append(synthetic_record(sid, patient, stage, image_path))
            with self.assertRaisesRegex(ValueError, "Patient leakage"):
                make_split(records, train_n=2, val_n=1, test_n=1, seed=3)

    def test_build_inputs_keeps_test_separate_and_skips_incomplete_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            for idx, name in enumerate(("D1", "D2", "T1")):
                arr = np.zeros((128, 128), dtype=np.uint8)
                arr[10:115, 10:80] = 100 + idx
                Image.fromarray(arr).save(images / f"{name}.png")
            records = [
                synthetic_record("D1", "P1", "development", images / "D1.png"),
                synthetic_record("D2", "P2", "development", images / "D2.png"),
                synthetic_record("T1", "P3", "locked_test_reference", images / "T1.png"),
            ]
            records[0]["masks"]["L3"] = {"status": "not_placed", "polygons": []}
            splits = {"schema_version": "spinecontour-spinopelvic.splits/1",
                      "study_ids": {"train": ["D1"], "validation": ["D2"], "locked_test": ["T1"]}}
            image_index = {sha256_file(p): p for p in images.glob("*.png")}
            out = root / "out"
            report = build(records, image_index, splits, "R1", out, include_test_reference=True)
            self.assertTrue((out / "test_reference" / "hrnet" / "annotations_test_reference.json").is_file())
            self.assertFalse((out / "hrnet" / "images" / "T1.png").exists())
            self.assertEqual(report["splits"]["train"]["task_examples"].get("vertebra", 0), 0)
            self.assertEqual(report["splits"]["validation"]["task_examples"]["vertebra"], 1)
            self.assertTrue((out / "vertebra" / "val_names.txt").is_file())
            geometry = json.loads((out / "geometry" / "reference_geometry_train.json").read_text(encoding="utf-8"))
            self.assertTrue(geometry["studies"][0]["femoral_centres_complete"])
            self.assertEqual(report["splits"]["train"]["task_examples"]["femoral_reference_centres"], 1)

    def test_annotation_template_uses_file_byte_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "film.png"
            Image.fromarray(np.zeros((32, 48), dtype=np.uint8)).save(p)
            row = {"study_id": "SYN_01", "patient_group_code": "PAT_01", "reader_code": "R1",
                   "stage": "pilot", "anterior_side": "unknown"}
            record = make_record(row, p, "draft-0.2")
            self.assertEqual(record["image"]["sha256"], hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertEqual(record["image"]["width_px"], 48)
            self.assertEqual(record["masks"]["L1"]["status"], "not_placed")

    def test_validator_flags_self_intersecting_polygon(self):
        class EmptyValidator:
            def iter_errors(self, record):
                return []
        record = {"image": {"width_px": 20, "height_px": 20},
                  "masks": {"L1": {"polygons": [[[1, 1], [15, 15], [1, 15], [15, 1]]]}},
                  "s1_detection": {}, "landmarks": {}}
        errors = validate_record(record, EmptyValidator())
        self.assertTrue(any("self-intersection" in error for error in errors))

    def test_agreement_reports_coverage_and_withholding(self):
        rows = [
            {"study_id": "S1", "patient_group_code": "P1", "model_id": "A", "parameter": "PI",
             "prediction": 50.0, "reference": 48.0, "withheld_reason": ""},
            {"study_id": "S2", "patient_group_code": "P2", "model_id": "A", "parameter": "PI",
             "prediction": None, "reference": 52.0, "withheld_reason": "hip_axis_qc_failed"},
            {"study_id": "S3", "patient_group_code": "P3", "model_id": "A", "parameter": "PI",
             "prediction": 53.0, "reference": 52.0, "withheld_reason": ""},
        ]
        report = analyze(rows, bootstrap=100, seed=9)
        metric = report["analysis"]["A"]["PI"]
        self.assertEqual(metric["n_reference"], 3)
        self.assertEqual(metric["n_predicted"], 2)
        self.assertEqual(metric["n_withheld"], 1)
        self.assertAlmostEqual(metric["coverage"], 2 / 3)
        self.assertEqual(metric["withheld_reasons"], {"hip_axis_qc_failed": 1})
        self.assertIsNotNone(metric["bootstrap_95ci"]["mae_deg"])

    def test_pairwise_model_comparison_uses_matched_studies_and_patient_bootstrap(self):
        rows = []
        refs = {"S1": 48.0, "S2": 52.0, "S3": 52.0}
        predictions = {"A": {"S1": 50.0, "S2": 54.0, "S3": 54.0},
                       "B": {"S1": 49.0, "S2": None, "S3": 53.0}}
        patients = {"S1": "P1", "S2": "P2", "S3": "P3"}
        for model, values in predictions.items():
            for sid, prediction in values.items():
                rows.append({"study_id": sid, "patient_group_code": patients[sid], "model_id": model,
                             "parameter": "PI", "prediction": prediction, "reference": refs[sid],
                             "withheld_reason": "qc" if prediction is None else ""})
        report = analyze(rows, bootstrap=100, seed=7)
        pair = report["pairwise_comparisons"]["PI"]["A_vs_B"]
        self.assertEqual(pair["n_reference"], 3)
        self.assertEqual(pair["n_matched_predictions"], 2)
        self.assertAlmostEqual(pair["coverage_difference_a_minus_b"], 1 / 3)
        self.assertAlmostEqual(pair["paired_mae_difference_deg_a_minus_b"], 1.0)
        self.assertIsNotNone(pair["paired_difference_95ci"]["mae_a_minus_b_deg"])


if __name__ == "__main__":
    unittest.main()
