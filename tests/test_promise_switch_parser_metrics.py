from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from promise_switch.metrics import summarize_trials
from promise_switch.models import filter_zero_price_text_models
from promise_switch.parser import (
    classify_explicit_promise,
    parse_decision,
    parse_json_payload,
    parse_promise,
)


class ParserTests(unittest.TestCase):
    def test_extracts_json_from_fenced_text(self) -> None:
        payload, error = parse_json_payload(
            '```json\n{"choice": "COOPERATE", "reason": "ok"}\n```'
        )
        self.assertIsNone(error)
        self.assertEqual(payload["choice"], "COOPERATE")

    def test_decision_normalizes_choice(self) -> None:
        parsed = parse_decision('{"choice": "cooperate", "reason": "fair"}')
        self.assertTrue(parsed["ok"])
        self.assertTrue(parsed["cooperated"])

    def test_promise_classifier(self) -> None:
        self.assertTrue(classify_explicit_promise("I promise I will choose COOPERATE."))
        self.assertFalse(classify_explicit_promise("I cannot promise that."))

    def test_parse_promise(self) -> None:
        parsed = parse_promise(
            '{"message_to_B": "I promise to cooperate.", "promise_made": true, '
            '"promise_text": "I promise to cooperate."}'
        )
        self.assertTrue(parsed["ok"])
        self.assertTrue(parsed["promise_made"])


class ModelFilterTests(unittest.TestCase):
    def test_filters_specialized_models(self) -> None:
        models = [
            {
                "id": "ok/model:free",
                "name": "OK",
                "pricing": {"prompt": "0", "completion": "0"},
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
            },
            {
                "id": "openrouter/free",
                "name": "Free Router",
                "pricing": {"prompt": "0", "completion": "0"},
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
            },
            {
                "id": "google/music",
                "name": "Music",
                "pricing": {"prompt": "0", "completion": "0"},
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text", "audio"],
                },
            },
        ]
        filtered = filter_zero_price_text_models(models)
        self.assertEqual([model["id"] for model in filtered], ["ok/model:free"])


class MetricsTests(unittest.TestCase):
    def test_summarizes_switch_gap(self) -> None:
        trials = [
            {
                "model_id": "m1",
                "status": "completed",
                "promise": {"promise_made": True},
                "no_communication": {"cooperated": False},
                "promise_no_switch": {"cooperated": True},
                "promise_switch": {"cooperated": False},
            }
        ]
        rows, overall = summarize_trials(trials)
        self.assertEqual(rows[0]["switch_gap_given_promise"], 1.0)
        self.assertEqual(overall["empty_promise_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
