#!/usr/bin/env bash
# run_kappa.sh — one validation of the coding instrument, over all fifteen
# configurations in the estimates.
#
# Two things are being fixed at once. The coded subsections rested on eleven
# configurations while the transfer results rested on fifteen, and the renderer
# fed both coders an exchange with no speaker labels, because it read a `role`
# field the engine never wrote. The judge saw "?:" on every line and the human
# saw "party", while the codebook asked each of them to code promise_investor
# and promise_trustee separately. Those two are the only variables defined per
# role and the only two that failed validity, at 0.467 and 0.286, against 0.648
# to 0.781 for the three that need no role.
#
# So the earlier rounds are superseded and not extended. One sample, one judge
# pass, one kappa.
#
#   ./run_kappa.sh sample    1. draw 75 items over 15 configurations   free
#   ./run_kappa.sh judge     2. recode all 10,479 transcripts          ~$4, ~3h
#      (code by hand)        3. fill data/coding/kappa_round4/          ~2.5h
#   ./run_kappa.sh reread    4. second machine reading, optional       ~$4, ~3h
#   ./run_kappa.sh score     5. the kappa
#
#   ./run_kappa.sh status    where things stand, no API calls
#   ./run_kappa.sh screen    look for a judge outside the study, before step 4
#
# Steps 1 and 2 are independent, since the draw excludes what a human has
# already read and pays no attention to what a machine has coded. Running the
# judge first is fine.
#
# The judge stays gpt-oss-120b at temperature 0 with the frozen codebook. It is
# also a configuration under study, so it reads its own 700 transcripts among
# the rest, which the paper reports.
set -u
cd "$(dirname "$0")"

ROUND=4
SAMPLE="data/coding/kappa_round${ROUND}"
CODES="data/coding/judge_comm.jsonl"

need_key() {
    if [ -z "${NEBIUS_API_KEY:-}" ] && [ ! -f ../.env ]; then
        echo "NEBIUS_API_KEY is not set and ../.env does not exist."
        echo "export it on its own line first:"
        echo "    export NEBIUS_API_KEY=..."
        exit 1
    fi
}

# The 7,920 rows already in the file were coded from unlabelled transcripts, so
# none of them can be compared with a row coded after the repair. Resume keys on
# the coder and the transcript, so leaving them in place would skip almost the
# whole run. They are moved aside once, kept, and never silently dropped.
archive_old() {
    if [ -f "$CODES" ] && ! [ -f "$CODES.pre_rolefix" ]; then
        mv "$CODES" "$CODES.pre_rolefix"
        echo "moved $(wc -l < "$CODES.pre_rolefix" | tr -d ' ') rows coded before the"
        echo "speaker labels were repaired to $(basename "$CODES").pre_rolefix"
        echo
    fi
}

case "${1:-status}" in

status)
    python3 - <<'PY'
import csv, json, collections, sys
from pathlib import Path
sys.path.insert(0, ".")
import _transcripts as T


def read_form(path):
    """A form round-tripped through a spreadsheet comes back semicolon
    separated in a European locale, so the delimiter is detected."""
    raw = path.read_text(encoding="utf-8-sig")
    try:
        d = csv.Sniffer().sniff(raw[:4000], delimiters=",;\t").delimiter
    except csv.Error:
        d = ","
    return list(csv.DictReader(path.open(encoding="utf-8-sig"), delimiter=d))


total = collections.Counter(a for a, _ in T.communication(exclude={"qwen3-think"}))
p = Path("data/coding/judge_comm.jsonl")
by = collections.Counter()
if p.exists():
    for line in p.read_text().splitlines():
        if line.strip():
            j = json.loads(line)
            if j.get("coded_clean"):
                by[(j.get("coder"), j.get("model_key"))] += 1

print("1. sample")
form = Path("data/coding/kappa_round4/coding_form.csv")
if not form.exists():
    print("     not drawn, run ./run_kappa.sh sample")
else:
    rows = read_form(form)
    print(f"     {len(rows)} items drawn")

stale = p.exists() and not Path(str(p) + ".pre_rolefix").exists()
print("\n2. judge, all configurations in the estimates")
if stale:
    print("     the counts below are codes made BEFORE the speaker labels were")
    print("     repaired, so none of them is usable. ./run_kappa.sh judge moves")
    print("     them aside and recodes from zero.")
done = sum(by[("judge:judge-gptoss", a)] for a in total)
for a in sorted(total):
    n = by[("judge:judge-gptoss", a)]
    bar = "done" if n >= total[a] else ""
    print(f"     {a:<22} {n:>5} / {total[a]:<5} {bar}")
print(f"     {'TOTAL':<22} {done:>5} / {sum(total.values()):<5} "
      f"{done / max(1, sum(total.values())):.0%}")

print("\n3. human coding")
if form.exists():
    rows = read_form(form)
    filled = sum(1 for r in rows if any((r.get(v) or "").strip() for v in
                 ("explicit_agreement", "agreement_type", "equality_framing")))
    print(f"     {filled} / {len(rows)} items coded")

print("\n4. second machine reading, optional")
done2 = sum(by[("judge:judge-gptoss-b", a)] for a in total)
print(f"     {done2} / {sum(total.values())}")
PY
    ;;

sample)
    rm -rf "$SAMPLE"
    KAPPA_ROUND=$ROUND KAPPA_N="${KAPPA_N:-75}" python3 make_kappa_sample.py
    echo
    echo "read  $SAMPLE/README.md"
    echo "code  $SAMPLE/transcripts.md  ->  $SAMPLE/coding_form.csv"
    ;;

screen)
    need_key
    echo "looking for a judge outside the study that honours the schema."
    echo "if one passes, use it at step 4 with JUDGE_B=<key>, since a second"
    echo "instrument measures more than a reread by the first."
    python3 screen_judges.py
    ;;

judge)
    need_key
    archive_old
    echo "recoding every configuration in the estimates with judge-gptoss."
    echo "resumable, so an interrupted run restarts with the same command."
    MODELS_JUDGE=judge-gptoss MAX_TOKENS=3000 \
        nohup python3 run_judge.py > log_judge_rolefix.log 2>&1 &
    echo "pid $!  ->  log_judge_rolefix.log"
    sleep 8
    tail -6 log_judge_rolefix.log
    echo
    echo "watch with:  tail -f $(pwd)/log_judge_rolefix.log"
    echo "             ./run_kappa.sh status"
    ;;

reread)
    need_key
    echo "second reading with ${JUDGE_B:-judge-gptoss-b}."
    MODELS_JUDGE="${JUDGE_B:-judge-gptoss-b}" MAX_TOKENS=3000 \
        nohup python3 run_judge.py > log_judge_reread.log 2>&1 &
    echo "pid $!  ->  log_judge_reread.log"
    sleep 8
    tail -6 log_judge_reread.log
    ;;

score)
    python3 score_kappa.py --round $ROUND
    ;;

*)
    sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
