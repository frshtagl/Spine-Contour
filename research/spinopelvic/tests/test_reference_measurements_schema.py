from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schemas" / "reference_measurements.schema.json").read_text(encoding="utf-8"))
EXAMPLE = json.loads((ROOT / "schemas" / "reference_measurements.example.json").read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA)


class ReferenceMeasurementSchemaTests(unittest.TestCase):
    def test_synthetic_reference_is_valid(self):
        self.assertEqual(list(VALIDATOR.iter_errors(EXAMPLE)), [])

    def test_not_assessable_requires_null_not_zero(self):
        record = copy.deepcopy(EXAMPLE)
        record["measurements"]["PI"] = {"status": "not_assessable", "value_deg": 0, "reason": "Hip centres not independently visible"}
        self.assertTrue(list(VALIDATOR.iter_errors(record)))

    def test_measured_value_requires_no_reason(self):
        record = copy.deepcopy(EXAMPLE)
        record["measurements"]["PI"]["reason"] = "uncertain"
        self.assertTrue(list(VALIDATOR.iter_errors(record)))


if __name__ == "__main__":
    unittest.main()
