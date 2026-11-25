from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "analysis"))
from normalize_model_predictions import _apply_hrnet, _apply_s1  # noqa: E402
from validate_annotations import validate_record  # noqa: E402
from compute_measurements_from_annotations import _point  # noqa: E402

SCHEMA = json.loads((ROOT / "schemas" / "annotation.schema.json").read_text(encoding="utf-8"))
EXAMPLE = json.loads((ROOT / "schemas" / "annotation.example.json").read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA)


def model_template():
    record = copy.deepcopy(EXAMPLE)
    record["model"] = {"name": "synthetic-model", "version": "test-1"}
    for landmark in record["landmarks"].values():
        landmark.update({"status": "not_placed", "point": None, "model_point": None,
                         "model_score": None, "uncertainty": "not_assessable", "uncertainty_note": None})
    record["s1_detection"].update({"status": "not_placed", "bbox": None, "model_bbox": None, "model_score": None})
    return record


class PredictionNormalizerTests(unittest.TestCase):
    def test_hrnet_threshold_keeps_raw_score_but_withholds_low_score_point(self):
        record = model_template()
        result = _apply_hrnet(record, {"landmarks": {"L1_SA": [10, 12], "L1_SP": [20, 12]},
                                      "conf": {"L1_SA": 0.2, "L1_SP": 0.8}}, 0.5)
        self.assertEqual(result["accepted"], 1)
        self.assertEqual(result["rejected_below_score"], 1)
        self.assertIsNone(record["landmarks"]["L1_SA"]["model_point"])
        self.assertEqual(record["landmarks"]["L1_SA"]["model_score"], 0.2)
        self.assertEqual(record["landmarks"]["L1_SP"]["model_point"], [20.0, 12.0])
        self.assertEqual(list(VALIDATOR.iter_errors(record)), [])

    def test_s1_model_bbox_and_points_remain_separate_from_human_fields(self):
        record = model_template()
        result = _apply_s1(record, {"bbox": [10, 20, 80, 90], "keypoints": [[20, 30], [70, 30]], "score": 0.4}, 0.5)
        self.assertTrue(result["detected"])
        self.assertFalse(result["accepted_for_geometry"])
        self.assertEqual(record["s1_detection"]["model_bbox"], [10.0, 20.0, 80.0, 90.0])
        self.assertIsNone(record["s1_detection"]["bbox"])
        self.assertIsNone(record["landmarks"]["S1_SA"]["model_point"])
        self.assertEqual(record["landmarks"]["S1_SA"]["model_score"], 0.4)
        self.assertEqual(list(VALIDATOR.iter_errors(record)), [])

    def test_geometry_uses_model_point_only_after_score_gate(self):
        record = {"model": {"name": "model", "version": "1"},
                  "landmarks": {"L1_SA": {"status": "not_placed", "point": None,
                                            "model_point": [12.0, 15.0], "model_score": 0.4}}}
        self.assertIsNone(_point(record, "L1_SA", min_model_score=0.5))
        self.assertEqual(_point(record, "L1_SA", min_model_score=0.3), [12.0, 15.0])


if __name__ == "__main__":
    unittest.main()
