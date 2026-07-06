"""
test_reskin.py — verification of the re-skin / contamination-control setup.

Runs WITHOUT network or API keys (stub chat_fn), so it can gate every run:
    python test_reskin.py      (or: pytest test_reskin.py -q)

What is verified:
  1. Re-skinned prompts (all roles, all phases) contain ZERO canonical
     surface markers — the paper's claim "the re-skin removes every
     identifying marker" becomes a tested invariant, not an assertion.
  2. Economic invariance under parameter shift: payoff identities and the
     selfish SPE hold for every variant (endowment 12, multiplier 2, ...).
  3. End-to-end dyad with a stub model works under every variant and logs
     the variant's parameters faithfully.
  4. Recognition-probe scoring classifies obvious cases correctly.
  5. run_reskin_pilot.run_full_series runs the full three-treatment series
     (no_history -> social_history -> communication) per variant offline,
     and the social_history report for a re-skinned variant leaks no
     canonical markers.
"""

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


# --------------------------------------------------------------------------- #
# Stub model: always sends/returns 3, valid JSON                              #
# --------------------------------------------------------------------------- #
def stub_chat_fn(system: str, user: str) -> str:
    return '{"reasoning": "stub", "amount": 3}'


MCFG = ModelConfig(model="stub", temperature=1.0)


def _all_prompts(cfg: GameConfig, with_comm: bool = True) -> str:
    """Every piece of text an agent could ever see under this config."""
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


# --------------------------------------------------------------------------- #
# 1. Re-skinned prompts carry no canonical markers                            #
# --------------------------------------------------------------------------- #
def test_reskin_prompts_have_no_canonical_markers():
    for name in ("reskin", "reskin_shifted"):
        text = _all_prompts(RESKINS[name])
        # The neutral labels themselves must not collide with marker words.
        hits = find_markers(text)
        assert hits == [], (
            f"variant '{name}' leaks canonical markers: {hits}\n"
            f"--- offending prompt text ---\n{text}"
        )


def test_canonical_config_does_contain_markers():
    # Sanity check that the detector actually detects.
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


# --------------------------------------------------------------------------- #
# 2. Economic invariance under parameter shift                                #
# --------------------------------------------------------------------------- #
def test_payoff_identities_all_variants():
    for name, cfg in RESKINS.items():
        rec = run_dyad(stub_chat_fn, cfg, "test", f"{name}-0", MCFG)
        e, m = cfg.endowment, cfg.multiplier
        s, r = rec["sent"], rec["returned"]
        assert rec["tripled"] == s * m, name
        assert rec["investor_payoff"] == e - s + r, name
        assert rec["trustee_payoff"] == e + s * m - r, name
        # Joint surplus created equals (m-1)*sent — the game's whole point.
        total = rec["investor_payoff"] + rec["trustee_payoff"]
        assert total == 2 * e + (m - 1) * s, name


def test_spe_is_invariant():
    """If the investor sends 0, both keep exactly the endowment — in EVERY
    variant. This is why hypotheses transfer across re-skins."""
    def zero_chat_fn(system: str, user: str) -> str:
        return '{"reasoning": "keep everything", "amount": 0}'
    for name, cfg in RESKINS.items():
        rec = run_dyad(zero_chat_fn, cfg, "test", f"{name}-spe", MCFG)
        assert rec["investor_payoff"] == cfg.endowment, name
        assert rec["trustee_payoff"] == cfg.endowment, name


def test_trustee_bound_uses_multiplier():
    """Trustee return is clamped to [0, m*sent] with the VARIANT's m."""
    def greedy_chat_fn(system: str, user: str) -> str:
        return '{"reasoning": "give a lot", "amount": 999}'
    cfg = RESKINS["reskin_shifted"]           # m = 2
    rec = run_dyad(greedy_chat_fn, cfg, "test", "bound-0", MCFG)
    assert rec["returned"] == rec["sent"] * 2  # clamped to m*sent, m=2 not 3


# --------------------------------------------------------------------------- #
# 3. Faithful provenance logging per variant                                  #
# --------------------------------------------------------------------------- #
def test_variant_parameters_are_logged():
    cfg = RESKINS["reskin_shifted"]
    rec = run_dyad(stub_chat_fn, cfg, "reskin_shifted", "log-0", MCFG)
    assert rec["endowment"] == 12 and rec["multiplier"] == 2
    assert "Blue Player" in rec["investor_prompt"] or \
           "Blue Player" in rec["trustee_prompt"]


# --------------------------------------------------------------------------- #
# 4. Recognition probe scoring                                                #
# --------------------------------------------------------------------------- #
def test_recognition_prompt_contains_rules_verbatim():
    rules = _rules_block(RESKINS["reskin_shifted"])
    system, user = recognition_probe_prompt(rules)
    assert rules in user
    assert find_markers(user) == []           # probe itself must not leak


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
    """Approximate human-benchmark claims are flagged even without 5.16/4.66
    (pilot 2026-07-02: models cite '30-50%' style knowledge freely)."""
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
    """A model that says recognized=false but then names the trust game is
    still scored as recognizing the paradigm (conservative scoring)."""
    completion = ('{"recognized": false, "paradigm_name": "", "reference": "", '
                  '"known_results": "this resembles the trust game of Berg"}')
    s = score_recognition(completion)
    assert s["recognized_paradigm"] and s["named_study"]


def test_communication_strips_private_reasoning():
    """Agent isolation in the comm phase. A model that emits <think> blocks in
    its message (e.g. qwen3 via Groq, found 2026-07-04) must never leak that
    reasoning into the transcript the counterpart reads. Raw kept for
    provenance, sanitized message rendered."""
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
        assert "SECRET" in turn["raw"]          # provenance preserved
    # the decision prompts (which embed the rendered transcript) are clean too
    assert "SECRET" not in rec["investor_prompt"]
    assert "SECRET" not in rec["trustee_prompt"]
    # truncated (unclosed) think block yields no leak either
    from berg_trust_game import strip_private_reasoning
    assert strip_private_reasoning("<think>half a thought") == ""
    assert strip_private_reasoning("Hello <think>x</think> there") == "Hello  there"


# --------------------------------------------------------------------------- #
# 5. run_reskin_pilot's full three-treatment series (offline, stub chat_fn)   #
# --------------------------------------------------------------------------- #
def test_run_full_series_runs_three_treatments_offline():
    """run_full_series (run_reskin_pilot.py) must, for one variant, produce
    exactly the three treatments no_history/social_history/communication,
    with the social_history report built from that SAME variant's own clean
    no_history records — and for a re-skinned variant, that report (and every
    prompt in the series) must leak no canonical markers."""
    import tempfile
    from pathlib import Path
    from run_reskin_pilot import run_full_series

    name = "reskin_shifted"          # maximal-distance variant: surface + params
    cfg = RESKINS[name]

    with tempfile.TemporaryDirectory() as d:
        out_path = Path(d) / "reskin_series_test.jsonl"
        by_treatment = run_full_series(
            stub_chat_fn, cfg, MCFG, name,
            n_no=2, n_soc=2, n_comm=2, comm_rounds=1,
            out_path=out_path, sleep_between=0.0,
        )

    expected = {
        f"{name}_no_history", f"{name}_social_history", f"{name}_communication",
    }
    assert set(by_treatment.keys()) == expected

    for treatment_name, records in by_treatment.items():
        assert len(records) == 2, treatment_name
        for rec in records:
            assert rec["treatment"] == treatment_name

    # social_history dyads must actually have been shown a history report...
    soc_records = by_treatment[f"{name}_social_history"]
    for rec in soc_records:
        assert rec["history_shown"] is True
        # ...and that report (embedded in the investor prompt) must be
        # marker-free, since the variant is a re-skin.
        assert find_markers(rec["investor_prompt"]) == [], rec["investor_prompt"]

    # no_history / communication dyads must NOT have been shown history.
    for treatment_name in (f"{name}_no_history", f"{name}_communication"):
        for rec in by_treatment[treatment_name]:
            assert rec["history_shown"] is False

    # communication dyads actually ran the comm phase (1 round -> 2 messages).
    for rec in by_treatment[f"{name}_communication"]:
        assert rec["comm_rounds"] == 1
        assert len(rec["transcript"]) == 2

    # Every prompt anywhere in the series must be marker-free (re-skin holds
    # end-to-end, not just in the history report).
    for records in by_treatment.values():
        for rec in records:
            assert find_markers(rec["investor_prompt"]) == []
            assert find_markers(rec["trustee_prompt"]) == []


# --------------------------------------------------------------------------- #
# Plain-python runner (no pytest needed)                                      #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys, traceback
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
