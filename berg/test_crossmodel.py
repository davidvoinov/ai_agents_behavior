"""
test_crossmodel.py — offline verification of cross-model dyad support in
berg_trust_game.py (Investor and Trustee played by different models/chat_fns).

Runs WITHOUT network or API keys (stub chat_fns), so it can gate every run:
    python3 test_crossmodel.py      (or: pytest test_crossmodel.py -q)

What is verified:
  1. Role isolation: investor and trustee stubs are wired to the correct role,
     amounts/payoffs come out correctly, and investor_model/trustee_model are
     logged from two independent ModelConfigs.
  2. The trustee stub actually receives the trustee prompt (proves the engine
     did not accidentally call the investor's chat_fn for the trustee, or
     vice versa).
  3. Backward compatibility: run_dyad with only the single chat_fn/mcfg (no
     trustee_* params) behaves exactly as before — investor_model ==
     trustee_model, "model"/"temperature" fields unchanged.
  4. run_treatment passes trustee_chat_fn/trustee_mcfg through to every dyad.
  5. run_crossmodel_pilot.run_full_series runs the full three-treatment
     series (no_history -> social_history -> communication) for one
     cross-model cell offline, with the Investor/Trustee stubs kept distinct
     across every treatment.
"""

from __future__ import annotations

from berg_trust_game import GameConfig, ModelConfig, run_dyad, run_treatment


# --------------------------------------------------------------------------- #
# Stubs that record every prompt they receive (for role-isolation proof)      #
# --------------------------------------------------------------------------- #
class RecordingStub:
    """Callable chat_fn that always returns a fixed completion and remembers
    every (system, user) prompt pair it was called with."""

    def __init__(self, completion: str):
        self.completion = completion
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.completion


INVESTOR_COMPLETION = '{"reasoning": "inv", "amount": 4}'
TRUSTEE_COMPLETION = '{"reasoning": "tru", "amount": 6}'


def make_stubs():
    investor_stub = RecordingStub(INVESTOR_COMPLETION)
    trustee_stub = RecordingStub(TRUSTEE_COMPLETION)
    return investor_stub, trustee_stub


CFG = GameConfig()  # endowment=10, multiplier=3 (faithful Berg defaults)


# --------------------------------------------------------------------------- #
# 1. Role isolation + payoffs + per-role model logging                        #
# --------------------------------------------------------------------------- #
def test_cross_model_role_isolation_and_payoffs():
    investor_stub, trustee_stub = make_stubs()
    inv_mcfg = ModelConfig(model="investor-model-A", temperature=0.7)
    tru_mcfg = ModelConfig(model="trustee-model-B", temperature=0.9)

    rec = run_dyad(
        investor_stub, CFG, "test_cross", "cross-0", inv_mcfg,
        trustee_chat_fn=trustee_stub, trustee_mcfg=tru_mcfg,
    )

    assert rec["sent"] == 4
    assert rec["returned"] == 6
    assert rec["tripled"] == 12

    # endowment=10, multiplier=3: investor 10-4+6=12, trustee 10+12-6=16
    assert rec["investor_payoff"] == 12
    assert rec["trustee_payoff"] == 16

    assert rec["investor_model"] == "investor-model-A"
    assert rec["trustee_model"] == "trustee-model-B"
    assert rec["investor_temperature"] == 0.7
    assert rec["trustee_temperature"] == 0.9

    # Legacy fields still reflect the investor's mcfg (backward compat contract).
    assert rec["model"] == "investor-model-A"
    assert rec["temperature"] == 0.7


# --------------------------------------------------------------------------- #
# 2. Trustee stub was actually called with the trustee prompt                 #
# --------------------------------------------------------------------------- #
def test_trustee_stub_receives_trustee_prompt():
    investor_stub, trustee_stub = make_stubs()
    inv_mcfg = ModelConfig(model="investor-model-A")
    tru_mcfg = ModelConfig(model="trustee-model-B")

    run_dyad(
        investor_stub, CFG, "test_cross", "cross-1", inv_mcfg,
        trustee_chat_fn=trustee_stub, trustee_mcfg=tru_mcfg,
    )

    # Each stub must have been called exactly once (one decision call per role;
    # comm_rounds=0 so no communication-phase calls).
    assert len(investor_stub.calls) == 1
    assert len(trustee_stub.calls) == 1

    inv_system, inv_user = investor_stub.calls[0]
    tru_system, tru_user = trustee_stub.calls[0]

    # The investor never sees "return" framing / the sent amount fed back to it.
    assert "Investor" in inv_system

    # The trustee prompt proves it's the trustee: it mentions returning money
    # and reflects X = 4 (the investor's parsed amount, tripled by cfg.multiplier).
    assert "Trustee" in tru_system
    assert "return" in tru_user.lower()
    assert "X = 4" in tru_user

    # Cross-check: the investor stub was never handed the trustee's prompt content.
    assert "X = 4" not in inv_user


# --------------------------------------------------------------------------- #
# 3. Backward compatibility: single chat_fn behaves as before                 #
# --------------------------------------------------------------------------- #
def test_backward_compat_single_chat_fn():
    def stub_chat_fn(system: str, user: str) -> str:
        # Same completion regardless of role, mirroring test_reskin.py's stub.
        return '{"reasoning": "stub", "amount": 3}'

    mcfg = ModelConfig(model="only-model", temperature=1.0)
    rec = run_dyad(stub_chat_fn, CFG, "test_compat", "compat-0", mcfg)

    assert rec["sent"] == 3
    assert rec["returned"] == 3
    assert rec["investor_model"] == rec["trustee_model"] == "only-model"
    assert rec["investor_temperature"] == rec["trustee_temperature"] == 1.0
    assert rec["model"] == "only-model"
    assert rec["temperature"] == 1.0


# --------------------------------------------------------------------------- #
# 4. run_treatment passes trustee_chat_fn/trustee_mcfg through                #
# --------------------------------------------------------------------------- #
def test_run_treatment_passes_through_cross_model_params(tmp_path=None):
    import tempfile
    from pathlib import Path

    investor_stub, trustee_stub = make_stubs()
    inv_mcfg = ModelConfig(model="investor-model-A")
    tru_mcfg = ModelConfig(model="trustee-model-B")

    with tempfile.TemporaryDirectory() as d:
        out_path = Path(d) / "crossmodel_test.jsonl"
        records = run_treatment(
            investor_stub, CFG, inv_mcfg, "test_cross_treatment", 2, out_path,
            trustee_chat_fn=trustee_stub, trustee_mcfg=tru_mcfg,
        )

    assert len(records) == 2
    for rec in records:
        assert rec["investor_model"] == "investor-model-A"
        assert rec["trustee_model"] == "trustee-model-B"
        assert rec["sent"] == 4
        assert rec["returned"] == 6

    # Both dyads should have driven exactly one call each into both stubs.
    assert len(investor_stub.calls) == 2
    assert len(trustee_stub.calls) == 2


# --------------------------------------------------------------------------- #
# 5. run_crossmodel_pilot's full three-treatment series (offline, stubs)      #
# --------------------------------------------------------------------------- #
def test_run_full_series_runs_three_treatments_offline():
    """run_full_series (run_crossmodel_pilot.py) must, for one cross-model
    cell, produce exactly the three treatments no_history/social_history/
    communication, with the Investor and Trustee stubs kept distinct in
    EVERY treatment (never falling back to a single shared chat_fn), and the
    social_history report built from that same cell's own clean no_history
    records."""
    import tempfile
    from pathlib import Path
    from run_crossmodel_pilot import run_full_series

    investor_stub, trustee_stub = make_stubs()
    inv_mcfg = ModelConfig(model="investor-model-A")
    tru_mcfg = ModelConfig(model="trustee-model-B")
    cellname = "cross_test-cell"

    with tempfile.TemporaryDirectory() as d:
        out_path = Path(d) / "crossmodel_series_test.jsonl"
        by_treatment = run_full_series(
            investor_stub, inv_mcfg, trustee_stub, tru_mcfg, CFG, cellname,
            n_no=2, n_soc=2, n_comm=2, comm_rounds=1,
            out_path=out_path, sleep_between=0.0,
        )

    expected = {
        f"{cellname}_no_history", f"{cellname}_social_history",
        f"{cellname}_communication",
    }
    assert set(by_treatment.keys()) == expected

    for treatment_name, records in by_treatment.items():
        assert len(records) == 2, treatment_name
        for rec in records:
            assert rec["treatment"] == treatment_name
            # Cross-model identity must hold in EVERY treatment of the series.
            assert rec["investor_model"] == "investor-model-A"
            assert rec["trustee_model"] == "trustee-model-B"
            assert rec["sent"] == 4
            assert rec["returned"] == 6

    soc_records = by_treatment[f"{cellname}_social_history"]
    for rec in soc_records:
        assert rec["history_shown"] is True

    for treatment_name in (f"{cellname}_no_history", f"{cellname}_communication"):
        for rec in by_treatment[treatment_name]:
            assert rec["history_shown"] is False

    for rec in by_treatment[f"{cellname}_communication"]:
        assert rec["comm_rounds"] == 1
        assert len(rec["transcript"]) == 2

    # 6 dyads total x (1 investor call + 1 trustee call), plus 2 communication
    # dyads x 1 comm round x 1 message each per role = +2 calls per stub.
    assert len(investor_stub.calls) == 6 + 2
    assert len(trustee_stub.calls) == 6 + 2


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
