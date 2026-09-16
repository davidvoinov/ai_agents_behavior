from __future__ import annotations

from berg_trust_game import GameConfig, ModelConfig, run_dyad, run_treatment

class RecordingStub:

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

CFG = GameConfig()

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

    assert rec["investor_payoff"] == 12
    assert rec["trustee_payoff"] == 16

    assert rec["investor_model"] == "investor-model-A"
    assert rec["trustee_model"] == "trustee-model-B"
    assert rec["investor_temperature"] == 0.7
    assert rec["trustee_temperature"] == 0.9

    assert rec["model"] == "investor-model-A"
    assert rec["temperature"] == 0.7

def test_trustee_stub_receives_trustee_prompt():
    investor_stub, trustee_stub = make_stubs()
    inv_mcfg = ModelConfig(model="investor-model-A")
    tru_mcfg = ModelConfig(model="trustee-model-B")

    run_dyad(
        investor_stub, CFG, "test_cross", "cross-1", inv_mcfg,
        trustee_chat_fn=trustee_stub, trustee_mcfg=tru_mcfg,
    )

    assert len(investor_stub.calls) == 1
    assert len(trustee_stub.calls) == 1

    inv_system, inv_user = investor_stub.calls[0]
    tru_system, tru_user = trustee_stub.calls[0]

    assert "Investor" in inv_system

    assert "Trustee" in tru_system
    assert "return" in tru_user.lower()
    assert "X = 4" in tru_user

    assert "X = 4" not in inv_user

def test_backward_compat_single_chat_fn():
    def stub_chat_fn(system: str, user: str) -> str:
        return '{"reasoning": "stub", "amount": 3}'

    mcfg = ModelConfig(model="only-model", temperature=1.0)
    rec = run_dyad(stub_chat_fn, CFG, "test_compat", "compat-0", mcfg)

    assert rec["sent"] == 3
    assert rec["returned"] == 3
    assert rec["investor_model"] == rec["trustee_model"] == "only-model"
    assert rec["investor_temperature"] == rec["trustee_temperature"] == 1.0
    assert rec["model"] == "only-model"
    assert rec["temperature"] == 1.0

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

    assert len(investor_stub.calls) == 2
    assert len(trustee_stub.calls) == 2

def test_run_full_series_runs_three_treatments_offline():
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

    assert len(investor_stub.calls) == 6 + 2
    assert len(trustee_stub.calls) == 6 + 2

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
