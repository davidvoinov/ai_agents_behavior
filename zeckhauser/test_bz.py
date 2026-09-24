import json
import sys
from pathlib import Path

import bz_game as BZ

def check(name, cond):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}")
    if not cond:
        raise AssertionError(name)

def main():
    cfg = BZ.BZConfig()

    print("payoffs / money-neutral MAP:")
    check("money-neutral MAP = 2/7", abs(cfg.money_neutral_map() - 2/7) < 1e-9)
    check("neutral MAP ~ 0.286", abs(cfg.money_neutral_map() - 0.2857) < 1e-3)

    print("condition prompts isolate the right structure:")
    dp = BZ.dm_system_prompt(cfg, "DP")
    rdg = BZ.dm_system_prompt(cfg, "RDG")
    tg = BZ.dm_system_prompt(cfg, "TG")
    check("DP has no Recipient and no Trustee",
          "Recipient" not in dp and "Trustee" not in dp)
    check("DP shows sure 10 and gamble 15/8",
          "10" in dp and "15" in dp and "8" in dp)
    check("RDG names the Recipient and uses a random draw",
          "Recipient" in rdg and "random draw" in rdg and "Trustee" not in rdg)
    check("RDG shows the other party's 15 and 22",
          "15" in rdg and "22" in rdg)
    check("TG hands the decision to a choosing Trustee",
          "Trustee" in tg and "hands the decision" in tg)
    check("TG shows the betrayal payoff C=22 > H=15",
          "22" in tg and "15" in tg)
    check("TG MAP ask is about the Trustee's choice",
          "Trustee" in BZ.dm_user_prompt(cfg, "TG") and
          "Trustee" not in BZ.dm_user_prompt(cfg, "DP"))

    print("parse_map:")
    cases = [
        ('{"reasoning":"r","map":0.5}', 0.5, True),
        ('{"map":"40%"}', 0.40, True),
        ('{"map":50}', 0.50, True),
        ('I would need about 0.3 chance', 0.30, False),
        ('{"map":150}', 1.0, False),
    ]
    for txt, exp, exp_clean in cases:
        v, _, clean = BZ.parse_map(txt)
        check(f"map({txt[:24]!r}) -> {v} clean={clean}",
              v is not None and abs(v - exp) < 1e-6 and clean == exp_clean)
    v, _, clean = BZ.parse_map('{"reasoning":"no number"}')
    check("map with no number -> None", v is None and clean is False)

    print("parse_choice:")
    for txt, exp in [('{"choice":"good"}', "good"), ('{"choice":"bad"}', "bad"),
                     ('{"choice":"betray"}', "bad"), ('{"choice":"honor"}', "good")]:
        c, _, _ = BZ.parse_choice(txt)
        check(f"choice({txt}) -> {c}", c == exp)
    c, _, clean = BZ.parse_choice('{"choice":"maybe"}')
    check("ambiguous choice -> None", c is None and clean is False)
    c, _, clean = BZ.parse_choice("I will betray them")
    check("no-json 'betray' scraped -> bad", c == "bad" and clean is False)

    print("run_unit + run_condition + summarize (stub model):")
    mcfg = BZ.ModelConfig(model="stub")

    def stub(system, user):
        if "You are the Trustee" in system:
            return '{"reasoning":"keep more","choice":"bad"}'
        return '{"reasoning":"r","map":0.5}'

    rec = BZ.run_unit(stub, cfg, "DP", "u0", mcfg)
    check("DP unit: map=0.5, no trustee choice",
          rec["map"] == 0.5 and rec["trustee_choice"] is None)
    rec = BZ.run_unit(stub, cfg, "TG", "u0", mcfg)
    check("TG unit: map=0.5 and trustee chose bad",
          rec["map"] == 0.5 and rec["trustee_choice"] == "bad")

    out = Path("/tmp/_bz_test.jsonl")
    if out.exists():
        out.unlink()
    summ = {}
    for cond in BZ.CONDITIONS:
        recs = BZ.run_condition(stub, cfg, mcfg, cond, 5, out)
        summ[cond] = BZ.summarize(recs, cond)
    check("each condition summarized to mean MAP 0.5",
          all(abs(summ[c]["mean_map"] - 0.5) < 1e-9 for c in BZ.CONDITIONS))
    check("TG trustworthiness = 0 (all trustees chose bad)",
          summ["TG"].get("trustworthiness") == 0.0)
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    check("JSONL wrote 15 valid records (5 per condition)",
          len(lines) == 15 and all(lines))
    BZ.betrayal_decomposition(summ)
    out.unlink()

    print("choice-list: parse_st + infer_switch + run:")
    for txt, exp in [('{"choice":"T"}', "T"), ('{"choice":"S"}', "S"),
                     ('{"choice":"gamble"}', "T"), ('{"choice":"the sure option"}', "S")]:
        c, _, _ = BZ.parse_st(txt)
        check(f"st({txt}) -> {c}", c == exp)
    c, _, clean = BZ.parse_st('{"choice":"?"}')
    check("st ambiguous -> None", c is None and clean is False)

    grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    mp, mono, pat = BZ.infer_switch(grid, ["S", "S", "T", "T", "T"])
    check("S,S,T,T,T -> 0.25 single_switch",
          abs(mp - 0.25) < 1e-9 and mono and pat == "single_switch")
    mp, mono, pat = BZ.infer_switch(grid, ["S"] * 5)
    check("all S -> None all_S", mp is None and mono and pat == "all_S")
    mp, mono, pat = BZ.infer_switch(grid, ["T"] * 5)
    check("all T -> 0.1 all_T", mp == 0.1 and mono and pat == "all_T")
    mp, mono, pat = BZ.infer_switch(grid, ["S", "T", "S", "T", "T"])
    check("S,T,S,T,T -> not monotone", (not mono) and pat == "multiple_switch")

    import re as _re

    def stub_cl(system, user):
        if "You are the Trustee" in system:
            return '{"reasoning":"keep more","choice":"bad"}'
        m = _re.search(r"(\d+)%", user)
        p = (int(m.group(1)) / 100.0) if m else 0.0
        return '{"reasoning":"r","choice":"%s"}' % ("T" if p >= 0.30 else "S")

    g = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    rec = BZ.run_choice_list_unit(stub_cl, cfg, "TG", "c0", mcfg, g)
    check("choice-list infers MAP 0.275 (midpoint 0.25/0.30)",
          abs(rec["map"] - 0.275) < 1e-9 and rec["map_clean"]
          and rec["switch_pattern"] == "single_switch")
    check("choice-list logs the full grid and the trustee choice",
          len(rec["grid_choices"]) == len(g) and rec["trustee_choice"] == "bad")

    print("\nALL BZ CHECKS PASSED")

if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)
