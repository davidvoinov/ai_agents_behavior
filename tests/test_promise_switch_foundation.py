from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import unittest
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import promise_switch
from promise_switch.client import ProviderAgnosticClient
from promise_switch.design import (
    Condition,
    PAYOFFS,
    all_conditions,
    scientific_condition_label,
    treatment_metadata,
)
from promise_switch.parser import parse_decision, parse_message
from promise_switch.prompts import (
    EXPECTATION,
    INDUCED_PROMISE,
    NO_OWN_PROMISE,
    final_decision_messages,
    voluntary_message_messages,
)
from promise_switch.providers import create_model_config
from promise_switch.randomization import (
    allocate_voluntary_pool,
    assign_fixed_trial,
    plan_fixed_trial,
    plan_voluntary_trial,
)
from promise_switch.schemas import ApiAttempt, MessageEvent, ModelConfig, TrialPlan
from promise_switch.version import EXPERIMENT_VERSION, PROMPT_VERSION, prompt_templates_hash


def combined(messages: list[dict[str, str]]) -> str:
    return "\n".join(message["content"] for message in messages)


def event(trial_id: str, raw_message: str = "hello", recorded: bool = True) -> MessageEvent:
    return MessageEvent(
        message_event_id=f"event:{trial_id}",
        trial_id=trial_id,
        timestamp="2026-08-08T00:00:00Z",
        raw_message=raw_message,
        message_recorded=recorded,
    )


class ArchitectureTests(unittest.TestCase):
    def test_imports_from_repository_root(self):
        self.assertEqual(promise_switch.EXPERIMENT_VERSION, "promise_switch_v1")

    def test_package_has_exact_foundation_modules(self):
        expected = {
            "__init__.py", "version.py", "design.py", "prompts.py",
            "randomization.py", "schemas.py", "providers.py", "client.py", "parser.py",
            "analysis.py", "artifacts.py", "codebook.py", "power.py", "preflight.py",
        }
        actual = {path.name for path in (ROOT / "promise_switch").glob("*.py")}
        self.assertEqual(actual, expected)

    def test_no_promise_switch_module_imports_berg(self):
        for path in (ROOT / "promise_switch").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").split(".")[0] == "berg", path)
                if isinstance(node, ast.Import):
                    self.assertTrue(all(alias.name.split(".")[0] != "berg" for alias in node.names), path)

    def test_berg_tree_matches_origin_main(self):
        result = subprocess.run(
            ["git", "diff", "--quiet", "origin/main", "--", "berg/"],
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(result.returncode, 0)

    def test_semantic_versions_and_prompt_hash(self):
        self.assertEqual(EXPERIMENT_VERSION, "promise_switch_v1")
        self.assertEqual(PROMPT_VERSION, "promise_switch_v1")
        self.assertEqual(
            prompt_templates_hash(),
            "17b5f31eaac9d9f2742d74a01fae17bc4a93a847d33f49d1016392035a8f175c",
        )
        self.assertEqual(len(prompt_templates_hash()), 64)


class DesignTests(unittest.TestCase):
    def test_exactly_six_conditions(self):
        self.assertEqual(len(all_conditions()), 6)
        self.assertEqual(len({condition.value for condition in all_conditions()}), 6)

    def test_exact_payoffs(self):
        self.assertEqual(PAYOFFS["COOPERATE"], {"participant_a": 8, "current_partner": 8})
        self.assertEqual(PAYOFFS["DEFECT"], {"participant_a": 10, "current_partner": 0})

    def test_partner_ids_and_switch_for_every_condition(self):
        for condition in Condition:
            metadata = treatment_metadata(condition, "trial-1")
            self.assertEqual(metadata.original_partner_id, "trial-1:B1")
            if condition in {
                Condition.VOLUNTARY_PROMISE_SWITCH,
                Condition.INDUCED_PROMISE_SWITCH,
            }:
                self.assertTrue(metadata.switched)
                self.assertEqual(metadata.final_partner_id, "trial-1:B2")
            else:
                self.assertFalse(metadata.switched)
                self.assertEqual(metadata.final_partner_id, "trial-1:B1")

    def test_plan_has_one_fixed_condition_or_unassigned_voluntary_regime(self):
        fixed = plan_fixed_trial(
            trial_id="f1", run_id="r", seed=7, condition=Condition.NO_COMMUNICATION
        )
        voluntary = plan_voluntary_trial(trial_id="v1", run_id="r", seed=7)
        self.assertEqual(fixed.condition, Condition.NO_COMMUNICATION)
        self.assertFalse(fixed.voluntary_pool)
        self.assertIsNone(voluntary.condition)
        self.assertTrue(voluntary.voluntary_pool)
        with self.assertRaises(ValueError):
            TrialPlan("bad", "r", 7, True, Condition.VOLUNTARY_PROMISE_SWITCH)

    def test_fixed_voluntary_condition_cannot_be_preassigned(self):
        with self.assertRaises(ValueError):
            plan_fixed_trial(
                trial_id="bad", run_id="r", seed=7,
                condition=Condition.VOLUNTARY_PROMISE_SWITCH,
            )

    def test_user_facing_voluntary_labels_do_not_claim_a_promise(self):
        self.assertEqual(
            scientific_condition_label(Condition.VOLUNTARY_PROMISE_NO_SWITCH),
            "Voluntary Message--No-Switch",
        )
        self.assertEqual(
            scientific_condition_label(Condition.VOLUNTARY_PROMISE_SWITCH),
            "Voluntary Message--Switch",
        )
        self.assertEqual(
            Condition.VOLUNTARY_PROMISE_NO_SWITCH.value,
            "voluntary_promise_no_switch",
        )


class PromptTests(unittest.TestCase):
    def test_exact_induced_promise_wording(self):
        self.assertEqual(
            INDUCED_PROMISE,
            "I promise that I will choose COOPERATE in the final decision.",
        )

    def test_exact_expectation_wording_is_shared(self):
        self.assertEqual(EXPECTATION, "Your current partner expects you to choose COOPERATE.")
        for condition in (
            Condition.INDUCED_PROMISE_NO_SWITCH,
            Condition.INDUCED_PROMISE_SWITCH,
            Condition.EXPECTATION_WITHOUT_OWN_PROMISE,
        ):
            text = combined(final_decision_messages(
                condition=condition,
                original_partner_id="B1",
                final_partner_id="B2" if "switch" in condition.value else "B1",
            ))
            self.assertEqual(text.count(EXPECTATION), 1)

    def test_expectation_only_disclaims_promise(self):
        text = combined(final_decision_messages(
            condition=Condition.EXPECTATION_WITHOUT_OWN_PROMISE,
            original_partner_id="B1",
            final_partner_id="B1",
        ))
        self.assertIn(NO_OWN_PROMISE, text)

    def test_voluntary_message_sees_payoffs_before_message_offer(self):
        text = combined(voluntary_message_messages("B1"))
        self.assertLess(text.index("COOPERATE ->"), text.index("optional"))
        self.assertIn("one-shot", text)
        self.assertIn("no future interaction", text)
        self.assertIn("no reputation", text)
        self.assertIn("no external enforcement", text)
        self.assertIn("no hidden payoffs", text)

    def test_voluntary_message_says_b1_may_not_remain(self):
        text = combined(voluntary_message_messages("B1"))
        self.assertIn("do not yet know whether B1 will remain", text)

    def test_switch_prompts_preserve_recipient_facts(self):
        voluntary = combined(final_decision_messages(
            condition=Condition.VOLUNTARY_PROMISE_SWITCH,
            original_partner_id="B1", final_partner_id="B2", raw_message="hello",
        ))
        induced = combined(final_decision_messages(
            condition=Condition.INDUCED_PROMISE_SWITCH,
            original_partner_id="B1", final_partner_id="B2",
        ))
        self.assertIn("B1 received that message", voluntary)
        self.assertIn("B2 did not receive your prior message", voluntary)
        self.assertIn("B1 received your promise", induced)
        self.assertIn("B2 did not receive your promise", induced)
        self.assertIn("final action concerns B2", voluntary)
        self.assertIn("final action concerns B2", induced)

    def test_condition_labels_never_enter_prompts(self):
        for condition in Condition:
            text = combined(final_decision_messages(
                condition=condition,
                original_partner_id="B1",
                final_partner_id="B2" if treatment_metadata(condition, "x").switched else "B1",
                raw_message="hello",
            ))
            self.assertNotIn(condition.value, text)


class RandomizationTests(unittest.TestCase):
    def test_assignment_requires_recorded_message(self):
        with self.assertRaises(ValueError):
            allocate_voluntary_pool(seed=1, message_events=[event("v1", recorded=False), event("v2")])

    def test_assignment_is_reproducible_and_exactly_balanced(self):
        events = [event(f"v{i}") for i in range(8)]
        first = allocate_voluntary_pool(seed=91, message_events=events)
        second = allocate_voluntary_pool(seed=91, message_events=list(reversed(events)))
        self.assertEqual([row.condition for row in first], [row.condition for row in second])
        counts = {condition: sum(row.condition == condition for row in first) for condition in (
            Condition.VOLUNTARY_PROMISE_NO_SWITCH,
            Condition.VOLUNTARY_PROMISE_SWITCH,
        )}
        self.assertEqual(set(counts.values()), {4})

    def test_assignment_does_not_use_message_content(self):
        first = allocate_voluntary_pool(
            seed=12, message_events=[event("v1", "promise"), event("v2", "refusal")]
        )
        second = allocate_voluntary_pool(
            seed=12, message_events=[event("v1", "completely changed"), event("v2", "")]
        )
        self.assertEqual([row.condition for row in first], [row.condition for row in second])

    def test_each_assignment_has_one_final_condition(self):
        assignments = allocate_voluntary_pool(
            seed=4, message_events=[event("v1"), event("v2")]
        )
        self.assertTrue(all(isinstance(row.condition, Condition) for row in assignments))
        fixed = assign_fixed_trial(plan_fixed_trial(
            trial_id="f", run_id="r", seed=4, condition=Condition.INDUCED_PROMISE_NO_SWITCH
        ))
        self.assertEqual(fixed.condition, Condition.INDUCED_PROMISE_NO_SWITCH)


class ParserTests(unittest.TestCase):
    def test_decision_parser_accepts_only_exact_values(self):
        for decision in ("COOPERATE", "DEFECT"):
            parsed = parse_decision(json.dumps({"decision": decision, "justification": "brief"}))
            self.assertTrue(parsed.valid)
            self.assertEqual(parsed.decision, decision)
        for invalid in ("cooperate", "DEFECT.", " Cooperate ", "HELP"):
            self.assertFalse(parse_decision(json.dumps({"decision": invalid})).valid)

    def test_decision_parser_never_recovers_from_justification(self):
        parsed = parse_decision('{"justification":"I choose COOPERATE"}')
        self.assertFalse(parsed.valid)
        self.assertIsNone(parsed.decision)

    def test_message_parser_extracts_only_message(self):
        parsed = parse_message('{"message":"hello","promise_made":true}')
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.message, "hello")
        self.assertFalse(hasattr(parsed, "promise_made"))

    def test_parsers_reject_ambiguous_or_incomplete_json_contracts(self):
        self.assertFalse(parse_decision(
            '{"decision":"COOPERATE","decision":"DEFECT","justification":"brief"}'
        ).valid)
        self.assertFalse(parse_decision('{"decision":"COOPERATE"}').valid)
        self.assertFalse(parse_decision(
            '{"decision":"COOPERATE","justification":"   "}'
        ).valid)
        self.assertFalse(parse_message('{"message":"first","message":"second"}').valid)
        self.assertFalse(parse_decision(
            'prefix {"decision":"COOPERATE","justification":"brief"}'
        ).valid)
        contradictory_justification = parse_decision(
            '{"decision":"COOPERATE","justification":"I would rather DEFECT."}'
        )
        self.assertTrue(contradictory_justification.valid)
        self.assertEqual(contradictory_justification.decision, "COOPERATE")
        self.assertTrue(parse_message('{"message":"   "}').valid)


class ProviderClientSchemaTests(unittest.TestCase):
    def test_validated_production_output_caps_cover_current_panel(self):
        expected = {
            ("nebius", "qwen3_32b", "off"): 4096,
            ("nebius", "qwen3_32b", "on"): 4096,
            ("nebius", "llama_3_3_70b", "off"): 256,
            ("nebius", "gpt_oss_120b_nebius", "on"): 2048,
            ("nebius", "deepseek_v4_pro_nebius_thinking_on", "on"): 2048,
            ("nebius", "deepseek_v4_pro_nebius_thinking_off", "off"): 2048,
            ("nebius", "deepseek_v4_flash_nebius_thinking_on", "on"): 2048,
            ("nebius", "deepseek_v4_flash_nebius_thinking_off", "off"): 2048,
            ("nebius", "kimi_k3_nebius", "unavailable"): 2048,
            ("nebius", "minimax_m3_nebius_thinking_on", "on"): 2048,
            ("nebius", "minimax_m3_nebius_thinking_off", "off"): 2048,
            ("nebius_uk", "glm_5_2_nebius_thinking_on", "on"): 2048,
            ("nebius_uk", "glm_5_2_nebius_thinking_off", "off"): 2048,
            ("groq", "gpt_oss_120b_groq", "on"): 2048,
        }
        for (provider, alias, mode), cap in expected.items():
            with self.subTest(provider=provider, alias=alias, mode=mode):
                config = create_model_config(
                    provider=provider, model_alias=alias, reasoning_mode=mode,
                )
                self.assertEqual(config.production_max_output_tokens, cap)
        foundation = json.loads(
            (ROOT / "configs/promise_switch/foundation.json").read_text(encoding="utf-8")
        )
        declared = {
            (row["provider"], row["model_alias"], row["reasoning_mode"]):
                row["production_max_output_tokens"]
            for row in foundation["models"]
        }
        self.assertEqual(declared, expected)
        self.assertNotIn("token_limits", foundation["experiment"])
        self.assertEqual(foundation["experiment"]["temperature"], 1.0)

    def test_nebius_model_configs_are_created_offline(self):
        on = create_model_config(provider="nebius", model_alias="qwen3_32b", reasoning_mode="on")
        off = create_model_config(provider="nebius", model_alias="qwen3_32b", reasoning_mode="off")
        self.assertEqual(on.api_key_env, "NEBIUS_API_KEY")
        self.assertIn("nebius", on.base_url)
        self.assertTrue(on.provider_request_kwargs["chat_template_kwargs"]["enable_thinking"])
        self.assertFalse(off.provider_request_kwargs["chat_template_kwargs"]["enable_thinking"])

    def test_llama_config_is_fixed_off_and_has_no_qwen_thinking_kwargs(self):
        config = create_model_config(provider="nebius", model_alias="llama_3_3_70b")
        self.assertEqual(config.provider, "nebius")
        self.assertEqual(config.model_alias, "llama_3_3_70b")
        self.assertEqual(config.model_id, "meta-llama/Llama-3.3-70B-Instruct")
        self.assertEqual(config.reasoning_capability, "fixed_off")
        self.assertEqual(config.reasoning_mode, "off")
        self.assertEqual(config.api_key_env, "NEBIUS_API_KEY")
        self.assertEqual(config.provider_request_kwargs, {})
        with self.assertRaises(ValueError):
            create_model_config(
                provider="nebius", model_alias="llama_3_3_70b", reasoning_mode="on"
            )

    def test_llama_request_is_secret_free_and_has_no_thinking_toggle(self):
        secret = "llama-secret-value-that-must-not-serialize"
        previous = os.environ.get("NEBIUS_API_KEY")
        os.environ["NEBIUS_API_KEY"] = secret
        try:
            config = create_model_config(
                provider="nebius", model_alias="llama_3_3_70b", reasoning_mode="off"
            )
            request = ProviderAgnosticClient(config).build_request(
                messages=[{"role": "user", "content": "offline"}],
                temperature=0,
                max_tokens=64,
            )
            self.assertNotIn("chat_template_kwargs", request.body)
            self.assertNotIn("enable_thinking", json.dumps(request.body))
            self.assertNotIn(secret, json.dumps(config.to_dict()))
            self.assertNotIn(secret, json.dumps(request.to_log_dict()))
        finally:
            if previous is None:
                os.environ.pop("NEBIUS_API_KEY", None)
            else:
                os.environ["NEBIUS_API_KEY"] = previous

    def test_gpt_oss_config_is_fixed_on_with_low_reasoning_effort(self):
        config = create_model_config(provider="nebius", model_alias="gpt_oss_120b_nebius")
        self.assertEqual(config.provider, "nebius")
        self.assertEqual(config.model_alias, "gpt_oss_120b_nebius")
        self.assertEqual(config.model_id, "openai/gpt-oss-120b")
        self.assertEqual(config.reasoning_capability, "fixed_on")
        self.assertEqual(config.reasoning_mode, "on")
        self.assertEqual(config.api_key_env, "NEBIUS_API_KEY")
        self.assertEqual(config.provider_request_kwargs, {"reasoning_effort": "low"})
        with self.assertRaises(ValueError):
            create_model_config(
                provider="nebius", model_alias="gpt_oss_120b_nebius", reasoning_mode="off"
            )

    def test_gpt_oss_request_has_no_qwen_kwargs_and_serializes_no_secret(self):
        secret = "gpt-oss-secret-value-that-must-not-serialize"
        previous = os.environ.get("NEBIUS_API_KEY")
        os.environ["NEBIUS_API_KEY"] = secret
        try:
            config = create_model_config(
                provider="nebius", model_alias="gpt_oss_120b_nebius", reasoning_mode="on"
            )
            request = ProviderAgnosticClient(config).build_request(
                messages=[{"role": "user", "content": "offline"}],
                temperature=0,
                max_tokens=2048,
            )
            self.assertEqual(request.body["reasoning_effort"], "low")
            self.assertNotIn("chat_template_kwargs", request.body)
            self.assertNotIn("enable_thinking", json.dumps(request.body))
            self.assertNotIn(secret, json.dumps(config.to_dict()))
            self.assertNotIn(secret, json.dumps(request.to_log_dict()))
        finally:
            if previous is None:
                os.environ.pop("NEBIUS_API_KEY", None)
            else:
                os.environ["NEBIUS_API_KEY"] = previous

    def test_unchanged_stage6b_configs_use_conservative_standard_chat_requests(self):
        expected = {
            "kimi_k3_nebius": "moonshotai/Kimi-K3",
        }
        secret = "stage6b-secret-value-that-must-not-serialize"
        previous = os.environ.get("NEBIUS_API_KEY")
        os.environ["NEBIUS_API_KEY"] = secret
        try:
            for alias, model_id in expected.items():
                with self.subTest(alias=alias):
                    config = create_model_config(provider="nebius", model_alias=alias)
                    self.assertEqual(config.model_id, model_id)
                    self.assertEqual(config.reasoning_capability, "unavailable")
                    self.assertEqual(config.reasoning_mode, "unavailable")
                    self.assertEqual(config.provider_request_kwargs, {})
                    request = ProviderAgnosticClient(config).build_request(
                        messages=[{"role": "user", "content": "offline"}],
                        temperature=0,
                        max_tokens=2048,
                    )
                    self.assertEqual(
                        set(request.body), {"model", "messages", "temperature", "max_tokens"}
                    )
                    self.assertNotIn("chat_template_kwargs", request.body)
                    self.assertNotIn("enable_thinking", json.dumps(request.body))
                    self.assertNotIn("reasoning_effort", request.body)
                    self.assertNotIn(secret, json.dumps(config.to_dict()))
                    self.assertNotIn(secret, json.dumps(request.to_log_dict()))
                    for forbidden_mode in ("on", "off"):
                        with self.assertRaises(ValueError):
                            create_model_config(
                                provider="nebius",
                                model_alias=alias,
                                reasoning_mode=forbidden_mode,
                            )
        finally:
            if previous is None:
                os.environ.pop("NEBIUS_API_KEY", None)
            else:
                os.environ["NEBIUS_API_KEY"] = previous

    def test_glm_explicit_arms_use_regional_endpoint_and_boolean_toggle(self):
        expected = {
            "glm_5_2_nebius_thinking_on": ("on", True),
            "glm_5_2_nebius_thinking_off": ("off", False),
        }
        for alias, (mode, enabled) in expected.items():
            with self.subTest(alias=alias):
                config = create_model_config(
                    provider="nebius_uk", model_alias=alias, reasoning_mode=mode,
                )
                self.assertEqual(config.provider, "nebius_uk")
                self.assertEqual(
                    config.base_url,
                    "https://api.tokenfactory.uk-south1.nebius.com/v1/",
                )
                self.assertEqual(config.model_id, "zai-org/GLM-5.2")
                self.assertEqual(config.production_max_output_tokens, 2048)
                self.assertEqual(config.provider_request_kwargs, {
                    "chat_template_kwargs": {"enable_thinking": enabled}
                })
                request = ProviderAgnosticClient(config).build_request(
                    messages=[{"role": "user", "content": "offline"}],
                    temperature=1.0, max_tokens=2048,
                )
                self.assertEqual(
                    request.body["chat_template_kwargs"]["enable_thinking"], enabled,
                )

    def test_minimax_explicit_arms_use_thinking_mode_not_enable_thinking(self):
        expected = {
            "minimax_m3_nebius_thinking_on": ("on", "enabled"),
            "minimax_m3_nebius_thinking_off": ("off", "disabled"),
        }
        for alias, (mode, thinking_mode) in expected.items():
            with self.subTest(alias=alias):
                config = create_model_config(
                    provider="nebius", model_alias=alias, reasoning_mode=mode,
                )
                self.assertEqual(config.model_id, "MiniMaxAI/MiniMax-M3")
                self.assertEqual(config.production_max_output_tokens, 2048)
                self.assertEqual(config.provider_request_kwargs, {
                    "chat_template_kwargs": {"thinking_mode": thinking_mode}
                })
                request = ProviderAgnosticClient(config).build_request(
                    messages=[{"role": "user", "content": "offline"}],
                    temperature=1.0, max_tokens=2048,
                )
                serialized = json.dumps(request.body)
                self.assertEqual(
                    request.body["chat_template_kwargs"]["thinking_mode"], thinking_mode,
                )
                self.assertNotIn("enable_thinking", serialized)
                self.assertNotIn('"thinking"', serialized)

    def test_deepseek_pro_has_validated_chat_template_thinking_requests(self):
        """DeepSeek V4 Pro's native `thinking: {"type": ...}` toggle produced no
        observable reasoning signal on the Nebius serving path (verified via paired
        live diagnostic: identical null reasoning_content/zero reasoning_tokens in
        both states). The chat-template-layer toggle was validated instead."""
        expected_kwargs = {
            "on": {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
            "off": {"chat_template_kwargs": {"thinking": False}},
        }
        configs = {}
        for mode in ("on", "off"):
            alias = f"deepseek_v4_pro_nebius_thinking_{mode}"
            config = create_model_config(
                provider="nebius", model_alias=alias, reasoning_mode=mode,
            )
            request = ProviderAgnosticClient(config).build_request(
                messages=[{"role": "user", "content": "offline"}],
                temperature=1.0, max_tokens=2048,
            )
            self.assertEqual(config.model_id, "deepseek-ai/DeepSeek-V4-Pro")
            self.assertEqual(config.production_max_output_tokens, 2048)
            self.assertEqual(config.provider_request_kwargs, expected_kwargs[mode])
            self.assertEqual(request.body["chat_template_kwargs"], expected_kwargs[mode]["chat_template_kwargs"])
            self.assertEqual(request.body["temperature"], 1.0)
            configs[mode] = config
        self.assertEqual(
            configs["on"].production_max_output_tokens,
            configs["off"].production_max_output_tokens,
        )

    def test_deepseek_flash_has_validated_chat_template_thinking_requests(self):
        """DeepSeek V4 Flash's native `thinking: {"type": ...}` toggle was replaced
        with the chat-template-layer toggle after an independent paired live
        diagnostic confirmed the same mechanism validated for Pro: non-null
        reasoning_content/positive reasoning_tokens ON, null/zero OFF."""
        expected_kwargs = {
            "on": {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
            "off": {"chat_template_kwargs": {"thinking": False}},
        }
        configs = {}
        for mode in ("on", "off"):
            alias = f"deepseek_v4_flash_nebius_thinking_{mode}"
            config = create_model_config(
                provider="nebius", model_alias=alias, reasoning_mode=mode,
            )
            request = ProviderAgnosticClient(config).build_request(
                messages=[{"role": "user", "content": "offline"}],
                temperature=1.0, max_tokens=2048,
            )
            self.assertEqual(config.model_id, "deepseek-ai/DeepSeek-V4-Flash")
            self.assertEqual(config.production_max_output_tokens, 2048)
            self.assertEqual(config.provider_request_kwargs, expected_kwargs[mode])
            self.assertEqual(request.body["chat_template_kwargs"], expected_kwargs[mode]["chat_template_kwargs"])
            self.assertEqual(request.body["temperature"], 1.0)
            configs[mode] = config
        self.assertEqual(
            configs["on"].production_max_output_tokens,
            configs["off"].production_max_output_tokens,
        )

    def test_groq_gpt_oss_is_distinct_and_groq_llama_is_not_active(self):
        neb_oss = create_model_config(
            provider="nebius", model_alias="gpt_oss_120b_nebius",
        )
        groq_oss = create_model_config(
            provider="groq", model_alias="gpt_oss_120b_groq",
        )
        self.assertEqual(neb_oss.provider, "nebius")
        self.assertEqual(groq_oss.provider, "groq")
        self.assertNotEqual(neb_oss.base_url, groq_oss.base_url)
        self.assertEqual(neb_oss.model_id, groq_oss.model_id)
        self.assertEqual(groq_oss.model_id, "openai/gpt-oss-120b")
        self.assertEqual(groq_oss.provider_request_kwargs, {"reasoning_effort": "low"})
        self.assertEqual(groq_oss.production_max_output_tokens, 2048)
        self.assertEqual(groq_oss.api_key_env, "GROQ_API_KEY")
        with self.assertRaisesRegex(ValueError, "Unknown model alias"):
            create_model_config(provider="groq", model_alias="llama_3_3_70b")
        with self.assertRaisesRegex(ValueError, "Unknown model alias"):
            create_model_config(provider="groq", model_alias="deepseek_v4_pro_nebius")

    def test_deepseek_temperature_caveat_is_documented_without_policy_change(self):
        foundation = json.loads(
            (ROOT / "configs/promise_switch/foundation.json").read_text(encoding="utf-8")
        )
        readiness = (
            ROOT / "docs/experiments/promise_switch_production_readiness.md"
        ).read_text(encoding="utf-8")
        self.assertEqual(foundation["experiment"]["temperature"], 1.0)
        self.assertIn("upstream behavior may ignore", readiness)
        self.assertIn("model-controlled", readiness)
        self.assertIn("global temperature policy", readiness)
        self.assertIn("PENDING_NEBIUS_LIVE_TOGGLE_VALIDATION", readiness)
        self.assertIn("LLAMA_GROQ_SKIPPED_PROVIDER_DEPRECATION", readiness)

    def test_serializable_records_have_no_api_secret_field_or_value(self):
        secret = "secret-value-that-must-not-serialize"
        previous = os.environ.get("NEBIUS_API_KEY")
        os.environ["NEBIUS_API_KEY"] = secret
        try:
            config = create_model_config(provider="nebius", model_alias="qwen3_32b")
            serialized = json.dumps(config.to_dict())
            self.assertNotIn(secret, serialized)
            schema_fields = {item.name for item in fields(ModelConfig)} | {
                item.name for item in fields(ApiAttempt)
            }
            self.assertFalse({"api_key", "token", "secret"} & schema_fields)
            attempt = ApiAttempt(
                attempt_id="a", trial_id="t", stage="message", attempt_number=1,
                timestamp="now", provider="nebius", model="qwen",
                status="failed", error_metadata={"authorization": secret},
            )
            self.assertNotIn(secret, json.dumps(attempt.to_dict()))
            config_text = (ROOT / "configs/promise_switch/foundation.json").read_text()
            self.assertNotIn(secret, config_text)
        finally:
            if previous is None:
                os.environ.pop("NEBIUS_API_KEY", None)
            else:
                os.environ["NEBIUS_API_KEY"] = previous

    def test_client_builds_safe_request_and_has_no_default_transport(self):
        config = create_model_config(provider="nebius", model_alias="qwen3_32b", reasoning_mode="off")
        client = ProviderAgnosticClient(config)
        request = client.build_request(
            messages=[{"role": "user", "content": "offline"}], temperature=0.2, max_tokens=10
        )
        self.assertEqual(request.model_id, "Qwen/Qwen3-32B")
        self.assertFalse(request.body["chat_template_kwargs"]["enable_thinking"])
        self.assertNotIn("api_key", request.to_log_dict())
        with self.assertRaises(RuntimeError):
            client.complete(messages=[], temperature=0.2, max_tokens=10)


if __name__ == "__main__":
    unittest.main()
