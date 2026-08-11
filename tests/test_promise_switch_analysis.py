from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from promise_switch.analysis import (
    Contrast, PRIMARY_CONFIRMATORY_CONTRASTS, analyze_by_model, binary_contrast, holm_adjust,
)
from promise_switch.artifacts import validate_run_directory
from promise_switch.codebook import (
    MessageCode, agreement_report, cohens_kappa, derived_outcomes, eligible_explicit_promiser,
)
from promise_switch.design import Condition
from promise_switch.execution.attrition import classify_decision_output, classify_transport
from promise_switch.parser import parse_decision
from promise_switch.schemas import NormalizedResponse
from promise_switch.version import PROMPT_VERSION, prompt_templates_hash

def row(condition: Condition, action: str, *, mode: str = "off") -> dict[str, str]:
    return {"provider": "test", "model": "model", "reasoning_mode": mode,
            "completion_status": "completed", "condition": condition.value,
            "final_action": action}


def complete_fixture(mode: str = "off") -> list[dict[str, str]]:
    rows = []
    for condition in Condition:
        rows.extend(row(condition, action, mode=mode)
                    for action in ("COOPERATE", "COOPERATE", "DEFECT", "DEFECT", "DEFECT"))
    return rows


class AnalysisTests(unittest.TestCase):
    def test_binary_contrast_reports_rates_ci_and_fisher_fallback(self) -> None:
        contrast = Contrast("test", Condition.INDUCED_PROMISE_NO_SWITCH, Condition.NO_COMMUNICATION)
        result = binary_contrast(complete_fixture(), contrast)
        self.assertAlmostEqual(result["treatment_rate"], 0.4)
        self.assertAlmostEqual(result["reference_rate"], 0.4)
        self.assertAlmostEqual(result["risk_difference"], 0)
        self.assertLess(result["risk_difference_ci_95"][0], 0)
        self.assertGreater(result["risk_difference_ci_95"][1], 0)
        self.assertEqual(result["test"], "fisher_exact_two_sided")

    def test_zero_full_and_large_mixed_binary_fixtures(self) -> None:
        contrast = Contrast("test", Condition.INDUCED_PROMISE_NO_SWITCH, Condition.NO_COMMUNICATION)
        extreme = ([row(Condition.INDUCED_PROMISE_NO_SWITCH, "COOPERATE") for _ in range(5)]
                   + [row(Condition.NO_COMMUNICATION, "DEFECT") for _ in range(5)])
        self.assertEqual(binary_contrast(extreme, contrast)["risk_difference"], 1.0)
        mixed = ([row(Condition.INDUCED_PROMISE_NO_SWITCH, "COOPERATE") for _ in range(10)]
                 + [row(Condition.INDUCED_PROMISE_NO_SWITCH, "DEFECT") for _ in range(10)]
                 + [row(Condition.NO_COMMUNICATION, "COOPERATE") for _ in range(10)]
                 + [row(Condition.NO_COMMUNICATION, "DEFECT") for _ in range(10)])
        self.assertEqual(binary_contrast(mixed, contrast)["test"], "pearson_chi_square_two_sided")

    def test_holm_is_monotone_in_sorted_p_order(self) -> None:
        adjusted = holm_adjust([{"p_value": .01}, {"p_value": .04}, {"p_value": .03}])
        ordered = sorted(adjusted, key=lambda result: result["p_value"])
        for actual, expected in zip([result["p_value_holm"] for result in ordered], [.03, .06, .06]):
            self.assertAlmostEqual(actual, expected)

    def test_analysis_never_pools_model_configurations(self) -> None:
        output = analyze_by_model(complete_fixture("off") + complete_fixture("on"))
        self.assertEqual(set(output), {("test", "model", "off"), ("test", "model", "on")})
        self.assertEqual(
            [item["name"] for item in output[("test", "model", "off")]["confirmatory_contrasts"]],
            [contrast.name for contrast in PRIMARY_CONFIRMATORY_CONTRASTS],
        )

    def test_confirmatory_h9_cannot_regress_to_voluntary_primary(self) -> None:
        h9 = PRIMARY_CONFIRMATORY_CONTRASTS[1]
        self.assertEqual(h9.name, "H9")
        self.assertEqual(h9.treatment, Condition.INDUCED_PROMISE_NO_SWITCH)
        self.assertEqual(h9.reference, Condition.INDUCED_PROMISE_SWITCH)
        output = analyze_by_model(complete_fixture())[("test", "model", "off")]
        self.assertEqual(
            output["secondary_analyses"]["voluntary_itt"]["classification"],
            "secondary_randomized_intention_to_treat",
        )
        self.assertEqual(
            output["secondary_analyses"]["voluntary_itt"]["treatment_label"],
            "Voluntary Message--No-Switch",
        )
        self.assertEqual(
            output["condition_labels"][Condition.VOLUNTARY_PROMISE_SWITCH.value],
            "Voluntary Message--Switch",
        )
        self.assertEqual(
            output["secondary_analyses"]["voluntary_explicit_promisers"]["classification"],
            "further_secondary_selected_subgroup",
        )

    def test_explicit_promiser_subset_is_model_specific_and_secondary(self) -> None:
        rows = complete_fixture()
        for index, item in enumerate(rows):
            item["trial_id"] = f"trial-{index}"
        voluntary_ids = {
            item["trial_id"] for item in rows
            if item["condition"] in {
                Condition.VOLUNTARY_PROMISE_NO_SWITCH.value,
                Condition.VOLUNTARY_PROMISE_SWITCH.value,
            }
        }
        output = analyze_by_model(
            rows,
            explicit_promiser_trial_ids_by_model={("test", "model", "off"): voluntary_ids},
        )[("test", "model", "off")]
        selected = output["secondary_analyses"]["voluntary_explicit_promisers"]
        self.assertEqual(selected["status"], "estimated")
        self.assertEqual(selected["classification"], "further_secondary_selected_subgroup")

    def test_incomplete_trials_are_excluded_and_empty_arm_fails(self) -> None:
        rows = complete_fixture()
        rows.append({**row(Condition.NO_COMMUNICATION, "COOPERATE"), "completion_status": "failed"})
        output = analyze_by_model(rows)
        self.assertEqual(
            output[("test", "model", "off")]["condition_counts"]["no_communication"]["analyzed_n"],
            5,
        )
        with self.assertRaisesRegex(ValueError, "empty arm"):
            binary_contrast([row(Condition.NO_COMMUNICATION, "COOPERATE")], PRIMARY_CONFIRMATORY_CONTRASTS[0])

    def test_codebook_validation_derivation_and_kappa(self) -> None:
        code = MessageCode(True, True, "promise", "COOPERATE", False, "", False, False)
        self.assertEqual(derived_outcomes(code, "DEFECT"), {"compliance": False, "promise_violation": True})
        self.assertAlmostEqual(cohens_kappa([True, True, False, False], [True, False, False, False]), 0.5)
        with self.assertRaisesRegex(ValueError, "explicit promise"):
            MessageCode(True, False, "promise", "COOPERATE", False, "", False, False)
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            MessageCode("yes", True, "promise", "COOPERATE", False, "", False, False)  # type: ignore[arg-type]

    def test_agreement_requires_matched_ids(self) -> None:
        base = {"explicit_promise": False, "commitment_present": False,
                "statement_type": "none", "promised_action": "NONE", "conditionality": False,
                "condition_text": "", "refusal_present": False, "ambiguous_response": False}
        report = agreement_report([{"trial_id": "a", **base}], [{"trial_id": "a", **base}])
        self.assertEqual(report["n_matched"], 1)
        self.assertTrue(all(value is None for value in report["kappa"].values()))
        self.assertNotIn("condition_text", report["kappa"])
        self.assertEqual(report["excluded_from_kappa"], ["condition_text"])
        with self.assertRaisesRegex(ValueError, "identical ids"):
            agreement_report([{"trial_id": "a", **base}], [{"trial_id": "b", **base}])
        with self.assertRaisesRegex(ValueError, "duplicate ids"):
            agreement_report(
                [{"trial_id": "a", **base}, {"trial_id": "a", **base}],
                [{"trial_id": "a", **base}],
            )

    def test_synthetic_message_edge_cases_remain_distinct(self) -> None:
        ambiguous = MessageCode(False, False, "suggestion", "COOPERATE", False, "", False, True)
        commitment = MessageCode(False, True, "intention", "COOPERATE", False, "", False, False)
        refusal = MessageCode(False, False, "none", "NONE", False, "", True, False)
        empty = MessageCode(False, False, "none", "NONE", False, "", False, False)
        self.assertIsNone(derived_outcomes(ambiguous, "COOPERATE")["compliance"])
        self.assertTrue(derived_outcomes(commitment, "COOPERATE")["compliance"])
        self.assertFalse(eligible_explicit_promiser(ambiguous))
        self.assertFalse(eligible_explicit_promiser(commitment))
        self.assertTrue(eligible_explicit_promiser(
            MessageCode(True, True, "promise", "COOPERATE", False, "", False, False)))
        self.assertTrue(refusal.refusal_present)
        self.assertEqual(empty.statement_type, "none")

    def test_synthetic_attrition_edges(self) -> None:
        timeout = classify_transport(NormalizedResponse(status="timeout", raw_response=""))
        quota = classify_transport(NormalizedResponse(
            status="error", raw_response="quota exhausted", http_status=429))
        truncation = classify_transport(NormalizedResponse(
            status="success", raw_response="", http_status=200, finish_reason="length"))
        contradictory = classify_decision_output(parse_decision("COOPERATE and DEFECT"))
        refusal = classify_decision_output(parse_decision("I refuse to decide"))
        invalid_json = classify_decision_output(parse_decision("not json"))
        self.assertEqual((timeout.cause, timeout.retryable), ("timeout", True))
        self.assertEqual((quota.cause, quota.quota_pause), ("quota_exhausted", True))
        self.assertEqual(truncation.cause, "token_truncation")
        self.assertEqual(contradictory.cause, "contradictory_action")
        self.assertEqual(refusal.cause, "refusal_to_decide")
        self.assertEqual(invalid_json.cause, "invalid_output_format")

    def test_derived_outcomes_reject_nonmechanical_action(self) -> None:
        code = MessageCode(True, True, "promise", "COOPERATE", False, "", False, False)
        with self.assertRaisesRegex(ValueError, "mechanical decision parser"):
            derived_outcomes(code, "cooperate")

    def test_analysis_reports_attrition_denominators(self) -> None:
        rows = complete_fixture()
        rows.append({**row(Condition.NO_COMMUNICATION, ""),
                     "completion_status": "failed", "failure_category": "behavioral"})
        counts = analyze_by_model(rows)[("test", "model", "off")]["condition_counts"]["no_communication"]
        self.assertEqual(counts["terminal_n"], 6)
        self.assertEqual(counts["completed_n"], 5)
        self.assertEqual(counts["analyzed_n"], 5)
        self.assertEqual(counts["behavioral_attrition"], 1)

    def test_duplicate_trial_ids_are_rejected(self) -> None:
        rows = complete_fixture()
        rows[0]["trial_id"] = "duplicate"
        rows[1]["trial_id"] = "duplicate"
        with self.assertRaisesRegex(ValueError, "duplicate trial_id"):
            analyze_by_model(rows)

    def test_invalid_and_legacy_runs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            manifest = {
                "execution_schema_version": "promise_switch_execution_v2",
                "experiment_version": "promise_switch_v1",
                "prompt_version": PROMPT_VERSION,
                "prompt_hash": prompt_templates_hash(),
                "dataset_role": "pilot",
                "dry_run": False,
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (run_dir / "trials.jsonl").write_text("", encoding="utf-8")
            self.assertEqual(validate_run_directory(run_dir), manifest)
            with self.assertRaisesRegex(ValueError, "dataset role"):
                validate_run_directory(run_dir, required_dataset_role="confirmatory")
            (run_dir / "INVALID_CONCURRENT_EXECUTION.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "INVALID_CONCURRENT_EXECUTION"):
                validate_run_directory(run_dir)
            validate_run_directory(run_dir, allow_invalid=True)
            (run_dir / "INVALID_CONCURRENT_EXECUTION.json").unlink()
            legacy = {
                "execution_schema_version": "promise_switch_execution_v1",
                "experiment_version": "promise_switch_v1",
            }
            (run_dir / "manifest.json").write_text(json.dumps(legacy), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "legacy or incompatible"):
                validate_run_directory(run_dir)
            self.assertEqual(validate_run_directory(run_dir, allow_legacy=True), legacy)


if __name__ == "__main__":
    unittest.main()
