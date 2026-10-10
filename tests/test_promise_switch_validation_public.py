from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.analyze_promise_switch_validation import (
    EXPECTED_IDS, FIELDS, PUBLIC_COLUMNS, PUBLIC_DIR, calculate_reliability,
    main, read_public_labels,
)


FORBIDDEN_COLUMNS = {
    "trial_id", "run_id", "source_model", "provider", "reasoning_mode",
    "treatment_assignment", "final_action", "message", "coder_id", "judge_id",
    "judge_model_id", "judge_provider", "judge_config_id", "api_metadata",
}


class PublicPromiseSwitchValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.human_path = PUBLIC_DIR / "human_labels_n130.csv"
        cls.judge_path = PUBLIC_DIR / "judge_labels_n130.csv"
        cls.human = read_public_labels(cls.human_path)
        cls.judge = read_public_labels(cls.judge_path)
        cls.summary = calculate_reliability(cls.human, cls.judge)

    def test_exact_public_ids_and_allowed_columns(self):
        for path, rows in ((self.human_path, self.human), (self.judge_path, self.judge)):
            with path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(tuple(csv.DictReader(handle).fieldnames or ()), PUBLIC_COLUMNS)
            identifiers = [row["public_item_id"] for row in rows]
            self.assertEqual(tuple(sorted(identifiers)), EXPECTED_IDS)
            self.assertEqual(len(identifiers), len(set(identifiers)))
            self.assertFalse(FORBIDDEN_COLUMNS & set(rows[0]))
        self.assertEqual(
            {row["public_item_id"] for row in self.human},
            {row["public_item_id"] for row in self.judge},
        )

    def test_no_private_identifiers_or_paths(self):
        for path in (self.human_path, self.judge_path):
            content = path.read_text(encoding="utf-8")
            for forbidden in (
                "/users/", "api_key", "trial_id", "run_id", "source_model",
                "provider", "reasoning_mode", "ps:", "private_", "judge_id",
            ):
                self.assertNotIn(forbidden, content.lower())

    def test_expected_reliability(self):
        expected = {
            "explicit_promise": (130, None),
            "commitment_present": (128, 0.900763358779),
            "statement_type": (121, 0.899085734000),
            "promised_action": (128, 0.901515151515),
            "conditionality": (123, 0.867346938776),
            "refusal_present": (130, None),
            "ambiguous_response": (128, 0.0),
        }
        self.assertEqual(tuple(row["variable"] for row in self.summary), FIELDS)
        for row in self.summary:
            agreement, kappa = expected[row["variable"]]
            self.assertEqual(row["raw_agreement_count"], agreement)
            if kappa is None:
                self.assertIsNone(row["cohens_kappa"])
                self.assertEqual(row["kappa_status"], "undefined_expected_agreement_one")
            else:
                self.assertAlmostEqual(row["cohens_kappa"], kappa, delta=1e-12)

    def test_summary_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.csv"
            second = Path(directory) / "second.csv"
            common = [
                "--human-labels", str(self.human_path),
                "--judge-labels", str(self.judge_path),
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main([*common, "--output", str(first)]), 0)
                self.assertEqual(main([*common, "--output", str(second)]), 0)
            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
