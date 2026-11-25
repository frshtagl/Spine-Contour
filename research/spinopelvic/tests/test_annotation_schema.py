from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from validate_annotations import validate_record  # noqa: E402

SCHEMA = json.loads((ROOT / "schemas" / "annotation.schema.json").read_text(encoding="utf-8"))
EXAMPLE = json.loads((ROOT / "schemas" / "annotation.example.json").read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA)


class AnnotationSchemaTests(unittest.TestCase):
    def test_synthetic_example_is_valid(self):
        self.assertEqual(validate_record(EXAMPLE, VALIDATOR), [])

    def test_out_of_bounds_landmark_is_rejected(self):
        record = copy.deepcopy(EXAMPLE)
        record["landmarks"]["FH1_CENTER"]["point"] = [1000, 760]
        errors = validate_record(record, VALIDATOR)
        self.assertTrue(any("outside the 1000x1000 image frame" in error for error in errors))

    def test_out_of_bounds_mask_vertex_is_rejected(self):
        record = copy.deepcopy(EXAMPLE)
        record["masks"]["L1"]["polygons"][0][0] = [-1, 100]
        errors = validate_record(record, VALIDATOR)
        self.assertTrue(any("masks.L1.polygons" in error for error in errors))

    def test_not_separable_head_cannot_have_a_fabricated_polygon(self):
        record = copy.deepcopy(EXAMPLE)
        record["masks"]["FEMORAL_HEAD_2"] = {"status": "not_separable", "polygons": []}
        record["landmarks"]["FH2_CENTER"] = {
            "status": "not_separable",
            "point": None,
            "uncertainty": "not_assessable"
        }
        self.assertEqual(validate_record(record, VALIDATOR), [])


if __name__ == "__main__":
    unittest.main()
