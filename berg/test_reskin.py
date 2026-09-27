from __future__ import annotations

import berg_trust_game as btg
from berg_trust_game import (
    GameConfig, ModelConfig, run_dyad,
    investor_system_prompt, trustee_system_prompt,
    investor_user_prompt, trustee_user_prompt, _rules_block,
)
from reskin import (
    RESKINS, CANONICAL_MARKERS, find_markers,
    recognition_probe_prompt, score_recognition,
)

def stub_chat_fn(system: str, user: str) -> str:
    return '{"reasoning": "stub", "amount": 3}'

MCFG = ModelConfig(model="stub", temperature=1.0)

def _all_prompts(cfg: GameConfig, with_comm: bool = True) -> str:
    transcript = [{"speaker": "investor", "label": cfg.investor_label,
                   "message": "hello"}]
    parts = [
        investor_system_prompt(cfg, with_comm),
        trustee_system_prompt(cfg, with_comm),
        investor_user_prompt(cfg, history_report=None, transcript=transcript),
        trustee_user_prompt(cfg, sent=3, tripled=3 * cfg.multiplier,
                            history_report=None, transcript=transcript),
        btg.comm_user_prompt(cfg, "investor", transcript),
        btg.comm_user_prompt(cfg, "trustee", transcript),
    ]
    return "\n\n".join(parts)

def test_reskin_prompts_have_no_canonical_markers():
    for name in ("reskin", "reskin_shifted"):
        text = _all_prompts(RESKINS[name])
        hits = find_markers(text)
        assert hits == [], (
            f"variant '{name}' leaks canonical markers: {hits}\n"
            f"--- offending prompt text ---\n{text}"
        )

def test_canonical_config_does_contain_markers():
    text = _all_prompts(RESKINS["canonical"])
    hits = find_markers(text)
    assert "investor" in hits and "trustee" in hits and "dollar" in hits

def test_history_report_is_marker_free_when_reskinned():
    cfg = RESKINS["reskin_shifted"]
    fake_records = [{
        "sent": 2, "returned": 1,
        "investor_parse_clean": True, "trustee_parse_clean": True,
    }]
    report = btg.build_history_report(fake_records, cfg)
    hits = find_markers(report)
    assert hits == [], f"history report leaks markers: {hits}\n{report}"

def test_payoff_identities_all_variants():
    for name, cfg in RESKINS.items():
        rec = run_dyad(stub_chat_fn, cfg, "test", f"{name}-0", MCFG)
        e, m = cfg.endowment, cfg.multiplier
        s, r = rec["sent"], rec["returned"]
        assert rec["tripled"] == s * m, name
        assert rec["investor_payoff"] == e - s + r, name
        assert rec["trustee_payoff"] == e + s * m - r, name
        total = rec["investor_payoff"] + rec["trustee_payoff"]
        assert total == 2 * e + (m - 1) * s, name

def test_spe_is_invariant():
    def zero_chat_fn(system: str, user: str) -> str:
        return '{"reasoning": "keep everything", "amount": 0}'
    for name, cfg in RESKINS.items():
        rec = run_dyad(zero_chat_fn, cfg, "test", f"{name}-spe", MCFG)
        assert rec["investor_payoff"] == cfg.endowment, name
        assert rec["trustee_payoff"] == cfg.endowment, name

def test_trustee_bound_uses_multiplier():
    def greedy_chat_fn(system: str, user: str) -> str:
        return '{"reasoning": "give a lot", "amount": 999}'
    cfg = RESKINS["reskin_shifted"]
    rec = run_dyad(greedy_chat_fn, cfg, "test", "bound-0", MCFG)
    assert rec["returned"] == rec["sent"] * 2

def test_variant_parameters_are_logged():
    cfg = RESKINS["reskin_shifted"]
    rec = run_dyad(stub_chat_fn, cfg, "reskin_shifted", "log-0", MCFG)
    assert rec["endowment"] == 12 and rec["multiplier"] == 2
    assert "Blue Player" in rec["investor_prompt"] or \
           "Blue Player" in rec["trustee_prompt"]

def test_recognition_prompt_contains_rules_verbatim():
    rules = _rules_block(RESKINS["reskin_shifted"])
    system, user = recognition_probe_prompt(rules)
    assert rules in user
    assert find_markers(user) == []

def test_score_recognition_positive():
    completion = ('{"recognized": true, "paradigm_name": "Investment game", '
                  '"reference": "Berg, Dickhaut & McCabe (1995)", '
                  '"known_results": "mean sent $5.16, returned $4.66"}')
    s = score_recognition(completion)
    assert s["recognized_paradigm"] and s["named_study"] and s["leaked_benchmark"]

def test_score_recognition_negative():
    completion = ('{"recognized": false, "paradigm_name": "", '
                  '"reference": "", "known_results": ""}')
    s = score_recognition(completion)
    assert not s["recognized_paradigm"]
    assert not s["named_study"]
    assert not s["leaked_benchmark"]

def test_score_recognition_approx_benchmark():
    pos = ('{"recognized": true, "paradigm_name": "Trust Game", "reference": '
           '"Berg et al. 1995", "known_results": "investors typically send '
           'about 30‑50% of their endowment"}')
    s = score_recognition(pos)
    assert s["approx_benchmark"] and not s["leaked_benchmark"]
    neg = ('{"recognized": false, "paradigm_name": "", "reference": "", '
           '"known_results": ""}')
    assert not score_recognition(neg)["approx_benchmark"]
    exact = ('{"recognized": true, "paradigm_name": "Trust Game", "reference": '
             '"Berg 1995", "known_results": "mean sent $5.16"}')
    assert score_recognition(exact)["leaked_benchmark"]

def test_score_recognition_keyword_overrides_selfreport():
    completion = ('{"recognized": false, "paradigm_name": "", "reference": "", '
                  '"known_results": "this resembles the trust game of Berg"}')
    s = score_recognition(completion)
    assert s["recognized_paradigm"] and s["named_study"]

def test_communication_strips_private_reasoning():
    def leaky_chat_fn(system, user):
        if "communication phase" in user and "do not output a JSON" in user:
            return "<think>SECRET plan to betray them</think>I promise to cooperate."
        return '{"reasoning": "d", "amount": 2}'
    rec = run_dyad(leaky_chat_fn, RESKINS["canonical"], "test", "leak-0", MCFG,
                   comm_rounds=1)
    assert len(rec["transcript"]) == 2
    for turn in rec["transcript"]:
        assert "SECRET" not in turn["message"] and "<think>" not in turn["message"]
        assert turn["message"] == "I promise to cooperate."
        assert "SECRET" in turn["raw"]
    assert "SECRET" not in rec["investor_prompt"]
    assert "SECRET" not in rec["trustee_prompt"]
    from berg_trust_game import strip_private_reasoning
    assert strip_private_reasoning("<think>half a thought") == ""
    assert strip_private_reasoning("Hello <think>x</think> there") == "Hello  there"
