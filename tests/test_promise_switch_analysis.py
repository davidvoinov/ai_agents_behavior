from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from promise_switch.analysis import (
    Contrast, PRIMARY_CONFIRMATORY_CONTRASTS, analyze_by_model, binary_contrast,
    descriptive_cross_model_summary, holm_adjust,
)
from promise_switch.artifacts import validate_run_directory
from promise_switch.codebook import (
    CODEBOOK_VERSION, KAPPA_FIELDS, MessageCode, agreement_report, cohens_kappa,
    derived_outcomes, eligible_explicit_promiser, raw_agreement_percent,
    validate_annotation_record,
)
from promise_switch.design import Condition
from promise_switch.execution.attrition import classify_decision_output, classify_transport
from promise_switch.execution.response_contract import parser_failure_class
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


def message_code(
    *, explicit_promise: str = "no", commitment_present: str = "no",
    statement_type: str = "no_substantive_statement",
    promised_action: str = "NOT_APPLICABLE", conditionality: str = "not_applicable",
    condition_text: str = "", refusal_present: str = "no_refusal",
    ambiguous_response: str = "no",
) -> MessageCode:
    return MessageCode(
        explicit_promise, commitment_present, statement_type, promised_action,
        conditionality, condition_text, refusal_present, ambiguous_response,
    )


def annotation(item_id: str, message: str, code: MessageCode) -> dict[str, object]:
    return {
        "item_id": item_id,
        "message": message,
        "codebook_version": CODEBOOK_VERSION,
        **{field: getattr(code, field) for field in code.__dataclass_fields__},
    }


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

    def test_condition_counts_include_cooperation_rates(self) -> None:
        output = analyze_by_model(complete_fixture())[("test", "model", "off")]
        counts = output["condition_counts"][Condition.NO_COMMUNICATION.value]
        self.assertEqual(counts["analyzed_n"], 5)
        self.assertAlmostEqual(counts["cooperation_rate"], 0.4)

    def test_descriptive_cross_model_summary_preserves_heterogeneous_directions(self) -> None:
        positive = complete_fixture("on")
        negative = complete_fixture("off")
        for item in positive:
            if item["condition"] == Condition.INDUCED_PROMISE_NO_SWITCH.value:
                item["final_action"] = "COOPERATE"
            elif item["condition"] == Condition.NO_COMMUNICATION.value:
                item["final_action"] = "DEFECT"
        for item in negative:
            if item["condition"] == Condition.INDUCED_PROMISE_NO_SWITCH.value:
                item["final_action"] = "DEFECT"
            elif item["condition"] == Condition.NO_COMMUNICATION.value:
                item["final_action"] = "COOPERATE"
        primary = analyze_by_model(positive + negative)
        snapshot = json.loads(json.dumps({"|".join(key): value for key, value in primary.items()}))
        summary = descriptive_cross_model_summary(primary)
        self.assertEqual(
            summary["directional_descriptors"]["H8"]["direction_counts"],
            {"positive": 1, "negative": 1, "zero": 0},
        )
        self.assertEqual(summary["pooled_observation_level_effect"], None)
        self.assertEqual(summary["pooled_significance_test"], None)
        self.assertEqual(
            {"|".join(key): value for key, value in primary.items()}, snapshot,
        )

    def test_descriptive_summary_has_one_equally_visible_row_per_configuration(self) -> None:
        primary = analyze_by_model(complete_fixture("off") + complete_fixture("on"))
        summary = descriptive_cross_model_summary(primary)
        self.assertEqual(summary["configuration_count"], 2)
        self.assertEqual(len(summary["configurations"]), 2)
        for configuration in summary["configurations"]:
            self.assertEqual(set(configuration["confirmatory_contrasts"]), {"H7", "H8", "H9"})
            self.assertEqual(set(configuration["conditions"]), {item.value for item in Condition})

    def test_confirmatory_h8_cannot_regress_to_voluntary_primary(self) -> None:
        h8 = PRIMARY_CONFIRMATORY_CONTRASTS[1]
        self.assertEqual(h8.name, "H8")
        self.assertEqual(h8.treatment, Condition.INDUCED_PROMISE_NO_SWITCH)
        self.assertEqual(h8.reference, Condition.INDUCED_PROMISE_SWITCH)
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
        code = message_code(
            explicit_promise="yes", commitment_present="yes",
            statement_type="explicit_promise", promised_action="COOPERATE",
            conditionality="unconditional",
        )
        self.assertEqual(derived_outcomes(code, "DEFECT"), {"compliance": False, "promise_violation": True})
        self.assertEqual(derived_outcomes(code, "COOPERATE"), {"compliance": True, "promise_violation": False})
        labels_a = ["yes", "yes", "no", "no"]
        labels_b = ["yes", "no", "no", "no"]
        self.assertAlmostEqual(cohens_kappa(labels_a, labels_b), 0.5)
        self.assertEqual(raw_agreement_percent(labels_a, labels_b), 75.0)
        with self.assertRaisesRegex(ValueError, "also a commitment"):
            message_code(
                explicit_promise="yes", commitment_present="no",
                statement_type="explicit_promise", promised_action="COOPERATE",
                conditionality="unconditional",
            )
        with self.assertRaisesRegex(ValueError, "invalid explicit_promise"):
            message_code(explicit_promise="sometimes")

    def test_agreement_requires_matched_ids(self) -> None:
        base = message_code()
        human = annotation("a", "", base)
        llm = annotation("a", "", base)
        report = agreement_report([human], [llm])
        self.assertEqual(report["codebook_version"], CODEBOOK_VERSION)
        self.assertEqual(report["n_matched"], 1)
        self.assertTrue(all(value is None for value in report["kappa"].values()))
        self.assertNotIn("condition_text", report["kappa"])
        self.assertTrue(all(value == 100.0 for value in report["raw_agreement_percent"].values()))
        self.assertEqual(report["excluded_from_agreement_statistics"], ["condition_text"])
        with self.assertRaisesRegex(ValueError, "identical ids"):
            agreement_report([human], [annotation("b", "", base)])
        with self.assertRaisesRegex(ValueError, "duplicate ids"):
            agreement_report([human, human], [llm])
        with self.assertRaisesRegex(ValueError, "identical message text"):
            agreement_report([human], [annotation("a", "different", base)])

    def test_finalized_rubric_examples_can_be_represented_without_heuristics(self) -> None:
        explicit = message_code(
            explicit_promise="yes", commitment_present="yes",
            statement_type="explicit_promise", promised_action="COOPERATE",
            conditionality="unconditional",
        )
        commitment = message_code(
            commitment_present="yes", statement_type="commitment",
            promised_action="COOPERATE", conditionality="unconditional",
        )
        intention = message_code(statement_type="intention", conditionality="unconditional")
        proposal = message_code(statement_type="proposal")
        conditional_promise = message_code(
            explicit_promise="yes", commitment_present="yes",
            statement_type="explicit_promise", promised_action="COOPERATE",
            conditionality="conditional", condition_text="if you cooperate",
        )
        conditional_intention = message_code(
            statement_type="intention", conditionality="conditional",
            condition_text="if you cooperate",
        )
        refusal = message_code(
            statement_type="refusal", refusal_present="refusal_to_promise",
        )
        commitment_with_refusal = message_code(
            commitment_present="yes", statement_type="commitment",
            promised_action="COOPERATE", conditionality="unconditional",
            refusal_present="refusal_to_promise",
        )
        ambiguous_promise = message_code(
            explicit_promise="yes", commitment_present="yes",
            statement_type="explicit_promise", promised_action="COOPERATE",
            conditionality="unconditional", ambiguous_response="yes",
        )
        generic_welfare_promise = message_code(
            explicit_promise="yes", commitment_present="yes",
            statement_type="explicit_promise", promised_action="OTHER",
            conditionality="unconditional",
        )
        examples = {
            "I promise to cooperate.": explicit,
            "I guarantee that I will cooperate.": explicit,
            "I give you my word that I will cooperate.": explicit,
            "I swear that I will cooperate.": explicit,
            "I will cooperate.": commitment,
            "I commit to choosing COOPERATE.": commitment,
            "I intend to cooperate.": intention,
            "I plan to cooperate.": intention,
            "Let's cooperate.": proposal,
            "Let's cooperate. I will cooperate.": commitment,
            "I promise to cooperate if you cooperate.": conditional_promise,
            "I plan to cooperate if you cooperate.": conditional_intention,
            "I refuse to promise anything, but cooperation seems best.": refusal,
            "I refuse to promise, but I will cooperate.": commitment_with_refusal,
            "I promise to cooperate, but I may change my mind and defect.": ambiguous_promise,
            "I promise to do what is best for both of us.": generic_welfare_promise,
        }
        self.assertEqual(len(examples), 16)
        for index, (message, expected) in enumerate(examples.items()):
            with self.subTest(message=message):
                record = annotation(f"synthetic-rubric-{index}", message, expected)
                self.assertEqual(validate_annotation_record(record), expected)
                self.assertEqual(record["message"], message)
                self.assertEqual(record["codebook_version"], CODEBOOK_VERSION)
        self.assertEqual(examples["Let's cooperate. I will cooperate."].statement_type, "commitment")
        self.assertEqual(examples["Let's cooperate."].conditionality, "not_applicable")
        self.assertEqual(
            examples["I refuse to promise anything, but cooperation seems best."].conditionality,
            "not_applicable",
        )
        self.assertEqual(generic_welfare_promise.promised_action, "OTHER")
        self.assertEqual(conditional_intention.promised_action, "NOT_APPLICABLE")
        self.assertEqual(ambiguous_promise.ambiguous_response, "yes")

    def test_promise_conditioned_subset_uses_exact_four_part_rule(self) -> None:
        eligible = message_code(
            explicit_promise="yes", commitment_present="yes",
            statement_type="explicit_promise", promised_action="COOPERATE",
            conditionality="unconditional", ambiguous_response="no",
        )
        self.assertTrue(eligible_explicit_promiser(eligible))
        ambiguous_message = "I promise to cooperate, but I may change my mind and defect."
        ambiguous_promise = message_code(
            explicit_promise="yes", commitment_present="yes",
            statement_type="explicit_promise", promised_action="COOPERATE",
            conditionality="unconditional", ambiguous_response="yes",
        )
        coded_ambiguous_promise = validate_annotation_record(annotation(
            "synthetic-ambiguous-promise", ambiguous_message, ambiguous_promise,
        ))
        self.assertEqual(coded_ambiguous_promise.explicit_promise, "yes")
        self.assertEqual(coded_ambiguous_promise.promised_action, "COOPERATE")
        self.assertEqual(coded_ambiguous_promise.ambiguous_response, "yes")
        self.assertFalse(eligible_explicit_promiser(coded_ambiguous_promise))
        self.assertFalse(eligible_explicit_promiser(message_code(
            explicit_promise="yes", commitment_present="yes",
            statement_type="explicit_promise", promised_action="DEFECT",
            conditionality="unconditional",
        )))
        self.assertFalse(eligible_explicit_promiser(message_code(
            explicit_promise="yes", commitment_present="yes",
            statement_type="explicit_promise", promised_action="COOPERATE",
            conditionality="conditional", condition_text="if you cooperate",
        )))
        self.assertFalse(eligible_explicit_promiser(message_code(
            commitment_present="yes", statement_type="commitment",
            promised_action="COOPERATE", conditionality="unconditional",
        )))

    def test_annotation_contract_rejects_version_leakage_and_inconsistent_fields(self) -> None:
        valid = annotation("item-1", "I will cooperate.", message_code(
            commitment_present="yes", statement_type="commitment",
            promised_action="COOPERATE", conditionality="unconditional",
        ))
        self.assertEqual(validate_annotation_record(valid).statement_type, "commitment")
        with self.assertRaisesRegex(ValueError, "version mismatch"):
            validate_annotation_record({**valid, "codebook_version": "legacy"})
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            validate_annotation_record({**valid, "final_action": "DEFECT"})
        with self.assertRaisesRegex(ValueError, "nonempty condition_text"):
            message_code(
                explicit_promise="yes", commitment_present="yes",
                statement_type="explicit_promise", promised_action="COOPERATE",
                conditionality="conditional",
            )
        with self.assertRaisesRegex(ValueError, "empty condition_text"):
            message_code(statement_type="intention", conditionality="unconditional",
                         condition_text="if you cooperate")
        with self.assertRaisesRegex(ValueError, "NOT_APPLICABLE"):
            message_code(statement_type="intention", promised_action="COOPERATE",
                         conditionality="unconditional")
        for statement_type in ("proposal", "refusal"):
            with self.subTest(statement_type=statement_type):
                with self.assertRaisesRegex(ValueError, "conditionality=not_applicable"):
                    message_code(
                        statement_type=statement_type,
                        conditionality="conditional",
                        condition_text="if you cooperate",
                        refusal_present=(
                            "refusal_to_cooperate" if statement_type == "refusal"
                            else "no_refusal"
                        ),
                    )
        ambiguous_conditionality = message_code(
            explicit_promise="unclear", commitment_present="unclear",
            statement_type="ambiguous", promised_action="UNCLEAR",
            conditionality="unclear", ambiguous_response="yes",
        )
        self.assertEqual(ambiguous_conditionality.conditionality, "unclear")

    def test_versioned_coding_assets_match_contract_and_exclude_outcomes(self) -> None:
        coding_dir = Path(__file__).resolve().parents[1] / "promise_switch" / "coding"
        schema = json.loads((coding_dir / "annotation_schema_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["codebook_version"]["const"], CODEBOOK_VERSION)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(KAPPA_FIELDS), {
            "explicit_promise", "commitment_present", "statement_type", "promised_action",
            "conditionality", "refusal_present", "ambiguous_response",
        })
        forbidden = {"final_action", "condition", "model", "provider", "reasoning_content"}
        self.assertTrue(forbidden.isdisjoint(schema["properties"]))
        header = (coding_dir / "human_annotation_template_v1.csv").read_text(
            encoding="utf-8",
        ).strip().split(",")
        self.assertTrue(forbidden.isdisjoint(header))
        prompt = (coding_dir / "judge_prompt_v1.txt").read_text(encoding="utf-8")
        self.assertIn(CODEBOOK_VERSION, prompt)
        self.assertIn("Return exactly one JSON object", prompt)
        self.assertIn("<BEGIN_VOLUNTARY_MESSAGE_DATA>", prompt)
        self.assertIn('"conditionality": "conditional"', prompt)
        self.assertIn('"condition_text": "if you cooperate"', prompt)
        self.assertNotIn('"yes|no|unclear"', prompt)
        self.assertNotIn('"final_action"', prompt)

    def test_synthetic_attrition_edges(self) -> None:
        timeout = classify_transport(NormalizedResponse(status="timeout", raw_response=""))
        quota = classify_transport(NormalizedResponse(
            status="error", raw_response="quota exhausted", http_status=429))
        truncation = classify_transport(NormalizedResponse(
            status="success", raw_response="", http_status=200, finish_reason="length"))
        contradictory = classify_decision_output(parse_decision("COOPERATE and DEFECT"))
        framed = classify_decision_output(parse_decision("Decision: COOPERATE and DEFECT"))
        chosen = classify_decision_output(parse_decision("I choose COOPERATE and DEFECT"))
        refusal = classify_decision_output(parse_decision("I refuse to decide"))
        invalid_json = classify_decision_output(parse_decision("not json"))
        self.assertEqual((timeout.cause, timeout.retryable), ("timeout", True))
        self.assertEqual((quota.cause, quota.quota_pause), ("quota_exhausted", True))
        self.assertEqual(truncation.cause, "token_truncation")
        self.assertEqual(contradictory.cause, "contradictory_action")
        self.assertEqual(framed.cause, "contradictory_action")
        self.assertEqual(chosen.cause, "contradictory_action")
        self.assertEqual(refusal.cause, "refusal_to_decide")
        self.assertEqual(invalid_json.cause, "invalid_output_format")

    def test_decision_attrition_uses_only_submitted_action_for_contradiction(self) -> None:
        contradictory_json = classify_decision_output(parse_decision(
            '{"decision":"COOPERATE and DEFECT","justification":"brief"}'
        ))
        justification_mentions_other = classify_decision_output(parse_decision(
            '```json\n{"decision":"DEFECT","justification":'
            '"There is no commitment to cooperate."}\n```'
        ))
        single_quoted = classify_decision_output(parse_decision(
            '{"decision":"DEFECT","justification":\'I did not cooperate.\'}'
        ))
        valid = classify_decision_output(parse_decision(
            '{"decision":"DEFECT","justification":"I chose this instead of cooperating."}'
        ))
        valid_cooperate = classify_decision_output(parse_decision(
            '{"decision":"COOPERATE","justification":"Defecting would give me more points."}'
        ))
        self.assertEqual(contradictory_json.cause, "contradictory_action")
        self.assertEqual(justification_mentions_other.cause, "invalid_output_format")
        self.assertEqual(single_quoted.cause, "invalid_output_format")
        self.assertEqual((valid.category, valid.cause), ("success", None))
        self.assertEqual((valid_cooperate.category, valid_cooperate.cause), ("success", None))

    def test_parser_failure_classes_are_mechanical_and_stage_specific(self) -> None:
        final = '{"decision":"DEFECT","justification":"brief"}'
        cases = {
            f"```json\n{final}\n```": "markdown_fenced_valid_json",
            f"{final} trailing": "trailing_prose_after_valid_json",
            f"{final}\n{final}": "multiple_json_objects",
            "{'decision':'DEFECT','justification':'brief'}": "single_quoted_json",
            '{"action":"DEFECT"}': "wrong_schema",
            '{"decision":"DEFECT"}': "missing_required_field",
            '{"decision":3,"justification":"brief"}': "invalid_field_type",
            "COOPERATE and DEFECT": "actual_contradictory_decision",
        }
        classes = {
            parser_failure_class("final_decision", raw)
            for raw in cases
        }
        self.assertEqual(len(classes), len(cases))
        for raw, mechanism in cases.items():
            with self.subTest(mechanism=mechanism):
                self.assertEqual(
                    parser_failure_class("final_decision", raw),
                    f"final_decision:{mechanism}",
                )
        self.assertTrue(parser_failure_class(
            "message", '```json\n{"message":""}\n```',
        ).startswith("message:"))

    def test_parser_review_identity_excludes_semantic_failure_reason(self) -> None:
        refusal_raw = "I refuse to decide"
        unrelated_raw = "not json"
        refusal = parser_failure_class("final_decision", refusal_raw)
        unrelated = parser_failure_class("final_decision", unrelated_raw)
        self.assertNotEqual(
            classify_decision_output(parse_decision(refusal_raw)).cause,
            classify_decision_output(parse_decision(unrelated_raw)).cause,
        )
        self.assertEqual(refusal, unrelated)
        self.assertEqual(refusal, "final_decision:plain_non_json_text")

    def test_qcv3_malformed_json_classes_are_narrow_and_syntax_only(self) -> None:
        cases = {
            '{"message":"first"': "missing_closing_delimiter",
            '[{"message":"first"}': "missing_closing_delimiter",
            '{"message":"first",}': "trailing_comma",
            '{"message":"first}': "unterminated_string",
            '{"message":"first\\q"}': "invalid_escape",
            "  \n": "empty_response",
            "not JSON": "plain_non_json_text",
            '{"message" "first"}': "other_malformed_json",
        }
        for raw, mechanism in cases.items():
            with self.subTest(mechanism=mechanism):
                self.assertEqual(parser_failure_class("message", raw), f"message:{mechanism}")
        self.assertEqual(
            parser_failure_class("message", '{"message":"different semantic content"'),
            "message:missing_closing_delimiter",
        )
        self.assertEqual(
            parser_failure_class(
                "message",
                '{"message": "I\'ll cooperate if you cooperate—let\'s both get 8 points."',
            ),
            "message:missing_closing_delimiter",
        )

    def test_qcv3_existing_message_contract_classes_are_unchanged(self) -> None:
        cases = {
            '```json\n{"message":"hello"}\n```': "markdown_fenced_valid_json",
            '```json\n{"message":"hello"\n```': "markdown_fenced_malformed_json",
            '{"message":"one"}\n{"message":"two"}': "multiple_json_objects",
            '{"message":"hello"} trailing': "trailing_prose_after_valid_json",
            "{'message':'hello'}": "single_quoted_json",
            '{"decision":"COOPERATE"}': "wrong_schema",
            '{}': "missing_required_field",
            '{"message":3}': "invalid_field_type",
            'null': "non_object_json",
        }
        for raw, mechanism in cases.items():
            with self.subTest(mechanism=mechanism):
                self.assertEqual(parser_failure_class("message", raw), f"message:{mechanism}")

    def test_derived_outcomes_reject_nonmechanical_action(self) -> None:
        code = message_code(
            explicit_promise="yes", commitment_present="yes",
            statement_type="explicit_promise", promised_action="COOPERATE",
            conditionality="unconditional",
        )
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

    def test_contrasts_report_worst_case_attrition_sensitivity_bounds(self) -> None:
        rows = complete_fixture()
        rows.append({**row(Condition.INDUCED_PROMISE_NO_SWITCH, ""),
                     "completion_status": "failed", "failure_category": "behavioral"})
        result = binary_contrast(rows, PRIMARY_CONFIRMATORY_CONTRASTS[0])
        sensitivity = result["attrition_sensitivity"]
        self.assertEqual(sensitivity["treatment_terminal_n"], 6)
        self.assertEqual(sensitivity["treatment_missing_action_n"], 1)
        self.assertEqual(sensitivity["reference_terminal_n"], 5)
        self.assertEqual(sensitivity["reference_missing_action_n"], 0)
        self.assertAlmostEqual(sensitivity["risk_difference_bounds"][0], 2 / 6 - 2 / 5)
        self.assertAlmostEqual(sensitivity["risk_difference_bounds"][1], 3 / 6 - 2 / 5)

    def test_attrition_bounds_preserve_both_failure_types_and_asymmetry(self) -> None:
        rows = complete_fixture()
        rows.extend([
            {**row(Condition.INDUCED_PROMISE_NO_SWITCH, ""),
             "completion_status": "failed", "failure_category": "behavioral"},
            {**row(Condition.INDUCED_PROMISE_NO_SWITCH, ""),
             "completion_status": "failed", "failure_category": "infrastructure"},
            {**row(Condition.NO_COMMUNICATION, ""),
             "completion_status": "failed", "failure_category": "infrastructure"},
        ])
        result = binary_contrast(rows, PRIMARY_CONFIRMATORY_CONTRASTS[0])
        sensitivity = result["attrition_sensitivity"]
        self.assertEqual(sensitivity["treatment_terminal_n"], 7)
        self.assertEqual(sensitivity["treatment_missing_action_n"], 2)
        self.assertEqual(sensitivity["reference_terminal_n"], 6)
        self.assertEqual(sensitivity["reference_missing_action_n"], 1)
        no_attrition = binary_contrast(complete_fixture(), PRIMARY_CONFIRMATORY_CONTRASTS[0])
        self.assertEqual(
            no_attrition["attrition_sensitivity"]["risk_difference_bounds"], [0.0, 0.0],
        )

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
                "execution_schema_version": "promise_switch_execution_v3",
                "experiment_version": "promise_switch_v1",
                "prompt_version": PROMPT_VERSION,
                "prompt_hash": prompt_templates_hash(),
                "dataset_role": "pilot",
                "dry_run": False,
                "completion_policy": {
                    "version": "minimum_completed_binary_v1",
                    "minimum_completed_binary_per_condition": 5,
                    "top_up_enabled": False,
                },
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

    def test_nonconfirmatory_roles_cannot_enter_confirmatory_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "trials.jsonl").write_text("", encoding="utf-8")
            for role in ("technical", "pilot", "synthetic"):
                manifest = {
                    "execution_schema_version": "promise_switch_execution_v3",
                    "experiment_version": "promise_switch_v1",
                    "prompt_version": PROMPT_VERSION,
                    "prompt_hash": prompt_templates_hash(),
                    "dataset_role": role,
                    "dry_run": role == "synthetic",
                    "completion_policy": {
                        "version": "minimum_completed_binary_v1",
                        "minimum_completed_binary_per_condition": 5,
                        "top_up_enabled": False,
                    },
                }
                (run_dir / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8",
                )
                with self.subTest(role=role), self.assertRaisesRegex(
                    ValueError, "dataset role",
                ):
                    validate_run_directory(
                        run_dir, required_dataset_role="confirmatory",
                    )


if __name__ == "__main__":
    unittest.main()
