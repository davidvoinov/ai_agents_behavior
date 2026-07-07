#!/usr/bin/env python3
"""
Run the free-stage Berg trust-game experiments.

The runner covers four cells:
1A same-model, no communication
1B cross-model, no communication
2A same-model, with communication
2B cross-model, with communication
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml
from dotenv import load_dotenv

try:
    from berg import berg_trust_game as B
except ImportError:
    import berg_trust_game as B

import run_experiment_1 as strict


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "free_full_experiments.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config is empty or invalid: {path}")
    return data


def resolve_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def index_models(models_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for model in models_cfg.get("models", []):
        name = model.get("name")
        if not name:
            raise ValueError("Every model entry must have a name.")
        out[name] = model
    return out


def check_model_ready(model_name: str, model: dict[str, Any]) -> None:
    required = ["provider", "model_family", "model_id", "api_key_env"]
    missing = [field for field in required if not model.get(field)]
    if missing:
        raise ValueError(f"{model_name} missing required fields: {', '.join(missing)}")
    key = os.environ.get(model["api_key_env"], "").strip()
    if not key or key.startswith("your-"):
        raise RuntimeError(f"Missing API key for {model_name}: {model['api_key_env']}")


def redact_error_text(error: Exception) -> str:
    text = str(error)
    for name, value in os.environ.items():
        if not value or len(value) < 8:
            continue
        if name.endswith("_API_KEY") or name.endswith("_TOKEN"):
            text = text.replace(value, "REDACTED")
    text = re.sub(r"([?&]key=)[^&\s\"']+", r"\1REDACTED", text)
    text = re.sub(r"user_[A-Za-z0-9]+", "user_REDACTED", text)
    return re.sub(r"org_[A-Za-z0-9]+", "org_REDACTED", text)


def make_gemini_chat_fn(model: dict[str, Any], temperature: float, max_tokens: int) -> B.ChatFn:
    api_key = os.environ[model["api_key_env"]]
    model_id = model["model_id"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
    min_call_interval = float(model.get("min_call_interval", 0.0) or 0.0)
    last_call = [0.0]

    def chat(system: str, user: str) -> str:
        if min_call_interval > 0:
            wait = min_call_interval - (time.time() - last_call[0])
            if wait > 0:
                time.sleep(wait)
        prompt = f"{system}\n\n{user}" if system else user
        response = requests.post(
            url,
            params={"key": api_key},
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            },
            timeout=60,
        )
        last_call[0] = time.time()
        if response.status_code >= 400:
            detail = response.text[:500].replace(api_key, "REDACTED")
            raise RuntimeError(f"Gemini API HTTP {response.status_code}: {detail}")
        data = response.json()
        return "".join(
            part.get("text", "")
            for candidate in data.get("candidates", [])
            for part in candidate.get("content", {}).get("parts", [])
        )

    return chat


def make_chat(model: dict[str, Any], temperature: float, max_tokens: int) -> tuple[B.ModelConfig, B.ChatFn]:
    provider_type = model.get("provider_type", "openai_compatible")
    mcfg = B.ModelConfig(
        model=model["model_id"],
        base_url=model.get("base_url"),
        api_key_env=model["api_key_env"],
        temperature=temperature,
        max_tokens=max_tokens,
        min_call_interval=float(model.get("min_call_interval", 0.0) or 0.0),
    )
    if provider_type == "gemini_rest":
        return mcfg, make_gemini_chat_fn(model, temperature, max_tokens)
    if provider_type == "openai_compatible":
        return mcfg, B.make_chat_fn(mcfg)
    raise ValueError(f"Unknown provider_type for {model['name']}: {provider_type}")


def build_same_pairs(model_names: list[str]) -> list[tuple[str, str]]:
    return [(name, name) for name in model_names]


def build_cross_pairs(model_names: list[str], reference: str, all_ordered: bool) -> list[tuple[str, str]]:
    if all_ordered:
        return [(a, b) for a in model_names for b in model_names if a != b]
    out = []
    for name in model_names:
        if name == reference:
            continue
        out.append((reference, name))
        out.append((name, reference))
    return out


def load_resume_state(path: Path) -> tuple[str | None, set[str], int]:
    if not path.exists():
        return None, set(), 0
    run_id = None
    completed: set[str] = set()
    rows = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows += 1
            rec = json.loads(line)
            run_id = run_id or rec.get("run_id")
            if rec.get("validation_success_A") and rec.get("validation_success_B"):
                completed.add(rec["dyad_id"])
    return run_id, completed, rows


def sanitize_message(raw: str) -> str:
    return B.strip_private_reasoning(raw).strip() or "(no message)"


def communication_prompt(game: B.GameConfig, role: str, transcript: list[dict[str, Any]]) -> str:
    return B.comm_user_prompt(game, role, transcript)


def run_communication(
    investor: B.Agent,
    trustee: B.Agent,
    investor_model: dict[str, Any],
    trustee_model: dict[str, Any],
    game: B.GameConfig,
    comm_rounds: int,
) -> list[dict[str, Any]]:
    transcript = []
    for _ in range(comm_rounds):
        for role, agent, label, model in (
            ("investor", investor, game.investor_label, investor_model),
            ("trustee", trustee, game.trustee_label, trustee_model),
        ):
            prompt = strict.apply_prompt_prefix(communication_prompt(game, role, transcript), model)
            raw = agent.respond(prompt).strip()
            transcript.append({
                "speaker": role,
                "label": label,
                "message": sanitize_message(raw),
                "raw": raw,
            })
    return transcript


def parse_decision(raw: str | None, model: dict[str, Any], field: str, lo: int, hi: int) -> dict[str, Any]:
    normalized = strict.normalize_response(raw, model)
    parsed = strict.parse_strict_decision(normalized["text"], field, lo, hi)
    parsed["normalized_response"] = normalized["text"]
    parsed["normalization_applied"] = normalized["applied"]
    parsed["normalization_rule"] = normalized["rule"]
    parsed["normalization_rules"] = normalized["rules"]
    return parsed


def run_dyad(
    *,
    run_id: str,
    experiment_id: str,
    treatment: str,
    pair_id: str,
    dyad_index: int,
    investor_model: dict[str, Any],
    trustee_model: dict[str, Any],
    investor_mcfg: B.ModelConfig,
    trustee_mcfg: B.ModelConfig,
    investor_chat: B.ChatFn,
    trustee_chat: B.ChatFn,
    game: B.GameConfig,
    comm_rounds: int,
) -> dict[str, Any]:
    with_comm = comm_rounds > 0
    investor_system = B.investor_system_prompt(game, with_comm=with_comm)
    trustee_system = B.trustee_system_prompt(game, with_comm=with_comm)
    investor = B.Agent("investor", investor_system, investor_chat)
    trustee = B.Agent("trustee", trustee_system, trustee_chat)

    transcript = run_communication(
        investor, trustee, investor_model, trustee_model, game, comm_rounds
    ) if with_comm else []

    transcript_text = None
    if transcript:
        transcript_text = "The communication phase is over. Full conversation:\n" + B.render_transcript(game, transcript)

    investor_prompt = strict.investor_decision_prompt(game)
    if transcript_text:
        investor_prompt = transcript_text + "\n\n" + investor_prompt
    investor_prompt = strict.apply_prompt_prefix(investor_prompt, investor_model)
    investor_raw = investor.respond(investor_prompt)
    investor_parsed = parse_decision(investor_raw, investor_model, "sent", 0, game.endowment)
    sent = investor_parsed["value"]

    if investor_parsed["validation_success"]:
        tripled = sent * game.multiplier
        trustee_prompt = strict.trustee_decision_prompt(game, sent, tripled)
        if transcript_text:
            trustee_prompt = transcript_text + "\n\n" + trustee_prompt
        trustee_prompt = strict.apply_prompt_prefix(trustee_prompt, trustee_model)
        trustee_raw = trustee.respond(trustee_prompt)
        trustee_parsed = parse_decision(trustee_raw, trustee_model, "returned", 0, tripled)
    else:
        tripled = None
        trustee_prompt = None
        trustee_raw = None
        trustee_parsed = {
            "value": None,
            "reasoning": "",
            "parsed_response": None,
            "parse_success": False,
            "validation_success": False,
            "error": "skipped because investor choice was invalid",
            "normalized_response": None,
            "normalization_applied": False,
            "normalization_rule": None,
            "normalization_rules": [],
        }

    returned = trustee_parsed["value"]
    valid = bool(investor_parsed["validation_success"] and trustee_parsed["validation_success"])
    payoff_a = game.endowment - sent + returned if valid else None
    payoff_b = game.endowment + tripled - returned if valid else None

    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "treatment": treatment,
        "pair_id": pair_id,
        "dyad_id": f"{pair_id}-{dyad_index:04d}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider_A": investor_model["provider"],
        "provider_B": trustee_model["provider"],
        "model_id_A": investor_model["model_id"],
        "model_id_B": trustee_model["model_id"],
        "model_family_A": investor_model["model_family"],
        "model_family_B": trustee_model["model_family"],
        "analysis_label_A": investor_model.get("analysis_label"),
        "analysis_label_B": trustee_model.get("analysis_label"),
        "endowment": game.endowment,
        "multiplier": game.multiplier,
        "sent": sent,
        "tripled_amount": tripled,
        "returned": returned,
        "payoff_A": payoff_a,
        "payoff_B": payoff_b,
        "joint_payoff": payoff_a + payoff_b if valid else None,
        "system_prompt_A": investor_system,
        "decision_prompt_A": investor_prompt,
        "raw_response_A": investor_raw,
        "normalized_response_A": investor_parsed["normalized_response"],
        "normalization_applied_A": investor_parsed["normalization_applied"],
        "normalization_rules_A": investor_parsed["normalization_rules"],
        "parsed_response_A": investor_parsed["parsed_response"],
        "system_prompt_B": trustee_system,
        "decision_prompt_B": trustee_prompt,
        "raw_response_B": trustee_raw,
        "normalized_response_B": trustee_parsed["normalized_response"],
        "normalization_applied_B": trustee_parsed["normalization_applied"],
        "normalization_rules_B": trustee_parsed["normalization_rules"],
        "parsed_response_B": trustee_parsed["parsed_response"],
        "parse_success_A": investor_parsed["parse_success"],
        "parse_success_B": trustee_parsed["parse_success"],
        "validation_success_A": investor_parsed["validation_success"],
        "validation_success_B": trustee_parsed["validation_success"],
        "parse_error_A": investor_parsed["error"],
        "parse_error_B": trustee_parsed["error"],
        "positive_trust": sent > 0 if sent is not None else None,
        "returned_zero_after_trust": sent > 0 and returned == 0 if valid else None,
        "returned_more_than_sent": sent > 0 and returned > sent if valid else None,
        "communication_rounds": comm_rounds,
        "communication_transcript": transcript,
    }


def failed_dyad_record(
    *,
    run_id: str,
    experiment_id: str,
    treatment: str,
    pair_id: str,
    dyad_index: int,
    investor_model: dict[str, Any],
    trustee_model: dict[str, Any],
    game: B.GameConfig,
    comm_rounds: int,
    error: Exception,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "treatment": treatment,
        "pair_id": pair_id,
        "dyad_id": f"{pair_id}-{dyad_index:04d}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider_A": investor_model["provider"],
        "provider_B": trustee_model["provider"],
        "model_id_A": investor_model["model_id"],
        "model_id_B": trustee_model["model_id"],
        "model_family_A": investor_model["model_family"],
        "model_family_B": trustee_model["model_family"],
        "analysis_label_A": investor_model.get("analysis_label"),
        "analysis_label_B": trustee_model.get("analysis_label"),
        "endowment": game.endowment,
        "multiplier": game.multiplier,
        "sent": None,
        "tripled_amount": None,
        "returned": None,
        "payoff_A": None,
        "payoff_B": None,
        "joint_payoff": None,
        "system_prompt_A": None,
        "decision_prompt_A": None,
        "raw_response_A": None,
        "normalized_response_A": None,
        "normalization_applied_A": False,
        "normalization_rules_A": [],
        "parsed_response_A": None,
        "system_prompt_B": None,
        "decision_prompt_B": None,
        "raw_response_B": None,
        "normalized_response_B": None,
        "normalization_applied_B": False,
        "normalization_rules_B": [],
        "parsed_response_B": None,
        "parse_success_A": False,
        "parse_success_B": False,
        "validation_success_A": False,
        "validation_success_B": False,
        "parse_error_A": None,
        "parse_error_B": None,
        "positive_trust": None,
        "returned_zero_after_trust": None,
        "returned_more_than_sent": None,
        "communication_rounds": comm_rounds,
        "communication_transcript": [],
        "run_error": redact_error_text(error),
        "run_error_type": type(error).__name__,
    }


def summarize(records: list[dict[str, Any]], label: str) -> None:
    clean = [r for r in records if r["validation_success_A"] and r["validation_success_B"]]
    print(f"\n=== {label} (n={len(clean)}/{len(records)} clean) ===")
    if not clean:
        return
    sent = [r["sent"] for r in clean]
    returned = [r["returned"] for r in clean]
    print(f"  mean sent     : {sum(sent) / len(sent):.2f}")
    print(f"  mean returned : {sum(returned) / len(returned):.2f}")
    print(f"  sent > 0      : {100 * sum(x > 0 for x in sent) / len(sent):.0f}%")


def main() -> None:
    load_dotenv(dotenv_path=ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n", type=int)
    parser.add_argument("--only-experiment", action="append", default=[])
    parser.add_argument("--all-cross", action="store_true")
    parser.add_argument("--resume-file")
    args = parser.parse_args()

    cfg_path = resolve_path(args.config)
    assert cfg_path is not None
    exp_cfg = load_yaml(cfg_path)
    models_cfg = load_yaml(resolve_path(exp_cfg["model_source"]["file"]))
    model_index = index_models(models_cfg)
    model_names = exp_cfg["models"]
    models = {name: model_index[name] for name in model_names}

    for name, model in models.items():
        check_model_ready(name, model)

    n = args.n or int(exp_cfg["sample"]["n_pairs_per_cell"])
    temp = float(exp_cfg["generation"]["temperature"])
    max_tokens = int(exp_cfg["generation"]["max_tokens"])
    comm_rounds = int(exp_cfg["generation"]["communication_rounds"])
    all_cross = args.all_cross or bool(exp_cfg["cross_model"].get("include_all_ordered_pairs"))

    same_pairs = build_same_pairs(model_names)
    cross_pairs = build_cross_pairs(model_names, exp_cfg["reference_model"], all_cross)
    experiments = [e for e in exp_cfg["experiments"] if e.get("enabled", False)]
    if args.only_experiment:
        allowed = set(args.only_experiment)
        experiments = [e for e in experiments if e["id"] in allowed]

    pair_counts = {
        "same_model": len(same_pairs),
        "cross_model": len(cross_pairs),
    }
    dyads = sum(pair_counts[e["pairing"]] * n for e in experiments)
    calls = 0
    for e in experiments:
        per_dyad = 2 + (2 * comm_rounds if e["communication"] else 0)
        calls += pair_counts[e["pairing"]] * n * per_dyad

    print("Models:")
    for name in model_names:
        print(f"- {name}: {models[name].get('analysis_label')}")
    print(f"n per cell: {n}")
    print(f"same pairs: {len(same_pairs)}")
    print(f"cross pairs: {len(cross_pairs)}")
    print(f"experiments: {', '.join(e['id'] for e in experiments)}")
    print(f"planned dyads: {dyads}")
    print(f"approx API calls: {calls}")

    if args.dry_run:
        return

    chat_cache: dict[str, tuple[B.ModelConfig, B.ChatFn]] = {}
    for name, model in models.items():
        mcfg, chat = make_chat(model, temp, max_tokens)
        chat_cache[name] = (mcfg, chat)

    if args.smoke:
        for name, (_, chat) in chat_cache.items():
            print(f"Smoke [{name}] ...", end=" ", flush=True)
            out = B.with_retry(lambda: chat("You output only JSON.", 'Return {"ok": true} and nothing else.'))
            print(" ".join(out.split())[:120])
        return

    out_dir = resolve_path(exp_cfg["output"]["directory"])
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    completed_dyads: set[str] = set()
    resumed_rows = 0
    if args.resume_file:
        out_path = resolve_path(args.resume_file)
        assert out_path is not None
        run_id, completed_dyads, resumed_rows = load_resume_state(out_path)
        run_id = run_id or out_path.stem
        print(f"Resuming {out_path}: {len(completed_dyads)} valid dyads already complete across {resumed_rows} rows.")
    else:
        run_id = f"{exp_cfg['output']['prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_path = out_dir / f"{run_id}.jsonl"
    game = B.GameConfig()

    with out_path.open("a", encoding="utf-8") as fh:
        for experiment in experiments:
            pairs = same_pairs if experiment["pairing"] == "same_model" else cross_pairs
            rounds = comm_rounds if experiment["communication"] else 0
            for investor_name, trustee_name in pairs:
                inv_model = models[investor_name]
                tru_model = models[trustee_name]
                inv_mcfg, inv_chat = chat_cache[investor_name]
                tru_mcfg, tru_chat = chat_cache[trustee_name]
                pair_id = f"{experiment['id']}__{investor_name}__{trustee_name}"
                records = []
                for i in range(n):
                    dyad_id = f"{pair_id}-{i:04d}"
                    if dyad_id in completed_dyads:
                        continue
                    try:
                        rec = run_dyad(
                            run_id=run_id,
                            experiment_id=experiment["id"],
                            treatment=experiment["id"],
                            pair_id=pair_id,
                            dyad_index=i,
                            investor_model=inv_model,
                            trustee_model=tru_model,
                            investor_mcfg=inv_mcfg,
                            trustee_mcfg=tru_mcfg,
                            investor_chat=inv_chat,
                            trustee_chat=tru_chat,
                            game=game,
                            comm_rounds=rounds,
                        )
                    except Exception as exc:
                        rec = failed_dyad_record(
                            run_id=run_id,
                            experiment_id=experiment["id"],
                            treatment=experiment["id"],
                            pair_id=pair_id,
                            dyad_index=i,
                            investor_model=inv_model,
                            trustee_model=tru_model,
                            game=game,
                            comm_rounds=rounds,
                            error=exc,
                        )
                        print(f"\nERROR {rec['dyad_id']}: {type(exc).__name__}: {rec['run_error']}", flush=True)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    records.append(rec)
                    if os.environ.get("SLEEP_BETWEEN"):
                        time.sleep(float(os.environ["SLEEP_BETWEEN"]))
                summarize(records, pair_id)

    print(f"\nDone. Raw records: {out_path}")


if __name__ == "__main__":
    main()
