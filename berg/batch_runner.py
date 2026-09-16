#!/usr/bin/env python3
"""
batch_runner.py — the confirmatory design through the Batch APIs, at half price.

Why this is not simply run_full_cycle with a different client. A dyad is
sequential: the trustee cannot be asked anything until the investor has decided,
and under the communication treatment four messages precede both decisions. A
batch holds only independent requests, so the run is restructured into six
waves, each of which is one batch per arm.

    1  communication, round 1, investor message
    2  communication, round 1, trustee message
    3  communication, round 2, investor message
    4  communication, round 2, trustee message
    5  investor decision          (every treatment)
    6  trustee decision           (every treatment)

Waves 1 to 4 carry only the communication cells, which is a quarter of the
design. The stimulus is built by the same functions the synchronous runner uses,
so the two paths cannot drift apart.

Only one process may hold an arm
--------------------------------
The ledger is the whole safety mechanism, and it only works if one process
writes it. Two runners started on the same arm each read a ledger with no entry
for the current wave, each submit their own batch, and the account pays twice.
That happened on 2026-09-11 for the two decision waves of gpt-5.6-nothink, at a
cost of about 0.40 dollars, which would have been 23 dollars on the Sonnet
thinking arm. An arm is therefore locked for the lifetime of the process, and a
second runner refuses to start while the lock holder is alive.

What happens if the run is interrupted
--------------------------------------
Money already committed is never spent twice, because a batch id is written to
the ledger BEFORE the first poll. A restart re-attaches to that id and reads its
results. Nothing is resubmitted while a live batch exists for the wave.

A balance that runs out is not fatal and not expensive. At submission the create
call fails outright and nothing is charged. Inside a running batch the affected
requests come back `errored`, and Anthropic does not bill errored, canceled or
expired requests, so the account pays only for what completed. Those custom ids
are collected and resubmitted as a follow-up batch of the same wave once the
balance is topped up. The design loses nothing, since every dyad carries its own
state and an unfinished one simply waits.

The one real deadline is the 24-hour expiry. A request the provider could not
reach in that window comes back `expired`, unbilled, and is retried the same way.

    python3 batch_runner.py --arms gpt-5.6-think --dyads 100
    python3 batch_runner.py --arms all --dry-run          # build, price, submit nothing
    python3 batch_runner.py --arms gpt-5.6-think          # rerunning resumes
    python3 batch_runner.py --status                      # what is in flight
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attrition as ATT
import berg_trust_game as B
import providers as P
import reskin as R
import run_full_cycle as RFC

ROOT = Path(__file__).resolve().parent / "data" / "batch"


def log(arm_dir: Path, msg: str) -> None:
    """Appended on every poll. A batch that is merely waiting looks exactly like
    a hung process from outside, and this is the difference."""
    arm_dir.mkdir(parents=True, exist_ok=True)
    with (arm_dir / "progress.log").open("a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")

ARMS_ALL = ["gpt-5.6-think", "gpt-5.6-nothink",
            "sonnet-5-think", "sonnet-5-nothink"]
SURFACES = list(R.RESKINS.keys())
CONDITIONS = ["no_history", "social_history_pos", "social_history_neg",
              "communication"]
COMM_ROUNDS = 2
POLL_SECONDS = 60

# Output budget per call. Thinking counts against it as a strict limit on both
# providers, so a reasoning arm that gets the same budget as a plain one loses
# the completions where the model thought longest. Claude Sonnet 5 at xhigh lost
# 7.7% of its dyads to this at 4000, every one of them on the investor call, 169
# of them returning nothing at all and the rest cut mid-sentence, concentrated in
# the unfavourable-report cell and the two longest surfaces. Nothing is paid for
# an unused budget, so the reasoning arms get room.
MAX_TOKENS = 4000
MAX_TOKENS_BY_ARM = {
    "sonnet-5-think": 16000,
    "gpt-5.6-think": 8000,
}


def budget(arm: str) -> int:
    return MAX_TOKENS_BY_ARM.get(arm, MAX_TOKENS)

# (name, round index, speaker). A decision wave carries a round of None.
WAVES = [("comm", 0, "investor"), ("comm", 0, "trustee"),
         ("comm", 1, "investor"), ("comm", 1, "trustee"),
         ("decide", None, "investor"), ("decide", None, "trustee")]


def wave_name(w) -> str:
    kind, rnd, who = w
    return f"{kind}{'' if rnd is None else rnd + 1}_{who}"


def safe_id(dyad_id: str) -> str:
    """A custom_id both providers accept.

    Anthropic enforces ^[a-zA-Z0-9_-]{1,64}$ on custom_id and rejects the whole
    batch otherwise. A dyad id is surface|condition|index, and the separator is
    the only character outside that set, so it becomes a hyphen. Surfaces and
    conditions carry no hyphens of their own, which keeps the change readable,
    and the caller maps back through an explicit dictionary rather than trusting
    the substitution to be reversible.
    """
    return dyad_id.replace("|", "-")


# --------------------------------------------------------------------------- #
# Backends                                                                    #
# --------------------------------------------------------------------------- #
class OpenAIBatch:
    """OpenAI Batch API. Requests go up as a file, results come back as one."""

    def __init__(self, spec: P.ModelSpec):
        from openai import OpenAI
        base_url, key_env, _ = P.PROVIDERS[spec.provider]
        key = os.environ.get(key_env, "").strip()
        if not key:
            raise SystemExit(f"set {key_env}")
        self.client = OpenAI(api_key=key,
                             **({"base_url": base_url} if base_url else {}))
        self.spec = spec

    def body(self, system: str, user: str) -> dict:
        b = {"model": self.spec.model,
             "messages": [{"role": "system", "content": system},
                          {"role": "user", "content": user}],
             "max_completion_tokens": budget(self.spec.key)}
        b.update(self.spec.extra_body or {})
        return b

    def submit(self, reqs: list[tuple[str, dict]], tmp: Path) -> str:
        tmp.write_text("\n".join(
            json.dumps({"custom_id": cid, "method": "POST",
                        "url": "/v1/chat/completions", "body": body})
            for cid, body in reqs) + "\n")
        up = self.client.files.create(file=tmp.open("rb"), purpose="batch")
        bat = self.client.batches.create(
            input_file_id=up.id, endpoint="/v1/chat/completions",
            completion_window="24h")
        return bat.id

    def status(self, bid: str) -> str:
        s = self.client.batches.retrieve(bid).status
        # completed / failed / expired / cancelled are terminal
        return "ended" if s in ("completed", "failed", "expired",
                                "cancelled") else "in_progress"

    def detail(self, bid: str) -> str:
        b = self.client.batches.retrieve(bid)
        c = getattr(b, "request_counts", None)
        n = (f"{c.completed}/{c.total} done, {c.failed} failed"
             if c else "counts not reported")
        return f"{b.status}, {n}"

    def results(self, bid: str) -> dict:
        bat = self.client.batches.retrieve(bid)
        out = {}
        for fid in (bat.output_file_id, bat.error_file_id):
            if not fid:
                continue
            for line in self.client.files.content(fid).text.splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                cid = r.get("custom_id")
                resp = (r.get("response") or {}).get("body") or {}
                if r.get("error") or not resp.get("choices"):
                    out[cid] = {"ok": False,
                                "error": json.dumps(r.get("error") or resp)[:300]}
                    continue
                ch = resp["choices"][0]
                u = resp.get("usage") or {}
                det = u.get("completion_tokens_details") or {}
                out[cid] = {"ok": True,
                            "text": ch["message"].get("content") or "",
                            "input_tokens": u.get("prompt_tokens"),
                            "output_tokens": u.get("completion_tokens"),
                            "reasoning_tokens": det.get("reasoning_tokens"),
                            "thinking_blocks": None,
                            "stop": ch.get("finish_reason")}
        return out


class AnthropicBatch:
    """Claude Message Batches. Errored, canceled and expired are not billed."""

    def __init__(self, spec: P.ModelSpec):
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise SystemExit("set ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=key)
        self.spec = spec

    def body(self, system: str, user: str) -> dict:
        eb = self.spec.extra_body or {}
        b = {"model": self.spec.model, "max_tokens": budget(self.spec.key),
             "system": system,
             "messages": [{"role": "user", "content": user}]}
        if eb.get("effort"):
            b["output_config"] = {"effort": eb["effort"]}
        if eb.get("thinking"):
            th = dict(eb["thinking"])
            if th.get("type") != "disabled":
                th.setdefault("display", "summarized")
            b["thinking"] = th
        # temperature is omitted on purpose: a non-default value is a 400 here
        # and the study runs at the model default.
        return b

    def submit(self, reqs: list[tuple[str, dict]], tmp: Path) -> str:
        bat = self.client.messages.batches.create(
            requests=[{"custom_id": cid, "params": body} for cid, body in reqs])
        return bat.id

    def status(self, bid: str) -> str:
        return self.client.messages.batches.retrieve(bid).processing_status

    def detail(self, bid: str) -> str:
        b = self.client.messages.batches.retrieve(bid)
        c = b.request_counts
        return (f"{b.processing_status}, {c.succeeded} succeeded, "
                f"{c.processing} processing, {c.errored} errored, "
                f"{c.expired} expired, {c.canceled} canceled")

    def results(self, bid: str) -> dict:
        out = {}
        for r in self.client.messages.batches.results(bid):
            kind = r.result.type
            if kind != "succeeded":
                # errored, canceled and expired are all unbilled and retryable
                out[r.custom_id] = {"ok": False, "error": kind,
                                    "retryable": kind in ("expired", "canceled",
                                                          "errored")}
                continue
            m = r.result.message
            text = "".join(b.text for b in m.content
                           if getattr(b, "type", "") == "text")
            blocks = [b for b in m.content if getattr(b, "type", "") in
                      ("thinking", "redacted_thinking")]
            out[r.custom_id] = {"ok": True, "text": text,
                                "input_tokens": m.usage.input_tokens,
                                "output_tokens": m.usage.output_tokens,
                                "reasoning_tokens": None,
                                "thinking_blocks": len(blocks),
                                "stop": m.stop_reason}
        return out


def backend(spec: P.ModelSpec):
    return AnthropicBatch(spec) if spec.api == "anthropic" else OpenAIBatch(spec)


# --------------------------------------------------------------------------- #
# Design and state                                                            #
# --------------------------------------------------------------------------- #
def plan(dyads: int) -> list[dict]:
    """Every dyad of the design, in the state it starts in."""
    out = []
    for surface in SURFACES:
        cfg = R.RESKINS[surface]
        for condition in CONDITIONS:
            _, cr, _ = RFC.condition_setup(cfg, condition, COMM_ROUNDS)
            for i in range(dyads):
                variant = i % RFC.N_HISTORY_VARIANTS
                hist, _, _ = RFC.condition_setup(cfg, condition, COMM_ROUNDS,
                                                 variant)
                out.append({
                    "dyad_id": f"{surface}|{condition}|{i:04d}",
                    "surface": surface, "condition": condition,
                    "treatment": RFC.ENGINE_TREATMENT[condition],
                    "history": hist, "comm_rounds": cr,
                    "stage": 0, "transcript": [], "sent": None,
                    "inv_raw": None, "tru_raw": None, "failed": None,
                })
                out[-1]["stage"] = next_stage(out[-1], 0)
    return out


def applies(d: dict, w) -> bool:
    """Does this wave exist for this dyad at all?

    The four message waves belong to the communication cells only, so a
    baseline dyad has to step over them instead of waiting in one.
    """
    kind, rnd, _ = w
    return d["comm_rounds"] > (rnd or 0) if kind == "comm" else True


def next_stage(d: dict, start: int) -> int:
    """The first wave at or after `start` that this dyad actually runs."""
    for i in range(start, len(WAVES)):
        if applies(d, WAVES[i]):
            return i
    return len(WAVES)


def needs(d: dict, w) -> bool:
    """Is this dyad due for this wave?"""
    return (not d["failed"] and applies(d, w)
            and d["stage"] == WAVES.index(w))


def build_request(d: dict, w, be) -> tuple[str, dict]:
    cfg = R.RESKINS[d["surface"]]
    with_comm = d["comm_rounds"] > 0
    kind, rnd, who = w
    if kind == "comm":
        system = (B.investor_system_prompt(cfg, True) if who == "investor"
                  else B.trustee_system_prompt(cfg, True))
        user = B.comm_user_prompt(cfg, who, d["transcript"])
    elif who == "investor":
        system = B.investor_system_prompt(cfg, with_comm)
        user = B.investor_user_prompt(cfg, d["history"],
                                      d["transcript"] or None)
    else:
        sent = d["sent"]
        system = B.trustee_system_prompt(cfg, with_comm)
        user = B.trustee_user_prompt(cfg, sent, sent * cfg.multiplier,
                                     d["history"], d["transcript"] or None)
    return d["dyad_id"], be.body(system, user)


def apply_result(d: dict, w, res: dict) -> None:
    cfg = R.RESKINS[d["surface"]]
    kind, rnd, who = w
    text = res["text"]
    if kind == "comm":
        msg = B.strip_private_reasoning(text).strip() or "(no message)"
        label = cfg.investor_label if who == "investor" else cfg.trustee_label
        d["transcript"].append({"speaker": who, "label": label,
                                "message": msg, "raw": text})
    elif who == "investor":
        sent, _, _ = B.parse_decision(text, 0, cfg.endowment)
        d["sent"] = sent
        d["inv_raw"] = text
    else:
        d["tru_raw"] = text
    d["stage"] = next_stage(d, WAVES.index(w) + 1)
    d.setdefault("usage", []).append(
        {k: res.get(k) for k in ("input_tokens", "output_tokens",
                                 "reasoning_tokens", "thinking_blocks")})


def finish(d: dict, spec: P.ModelSpec) -> dict:
    """Assemble the record, in the schema the synchronous runs already use."""
    cfg = R.RESKINS[d["surface"]]
    sent, inv_reason, inv_clean = B.parse_decision(d["inv_raw"] or "", 0,
                                                   cfg.endowment)
    tripled = sent * cfg.multiplier
    lo = B.min_return(cfg, tripled)
    returned, tru_reason, tru_clean = B.parse_decision(d["tru_raw"] or "", lo,
                                                       tripled)
    usage = d.get("usage", [])
    return {
        "dyad_id": d["dyad_id"], "treatment": d["treatment"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "model": spec.model, "model_key": spec.key, "temperature": 1.0,
        "investor_model": spec.model, "trustee_model": spec.model,
        "investor_temperature": 1.0, "trustee_temperature": 1.0,
        "surface": d["surface"], "condition": d["condition"],
        "history_shown": bool(d["history"]), "comm_rounds": d["comm_rounds"],
        "endowment": cfg.endowment, "multiplier": cfg.multiplier,
        "min_return_frac": cfg.min_return_frac,
        "sent": sent, "tripled": tripled, "returned": returned,
        "return_ratio": (returned / tripled) if tripled > 0 else None,
        "investor_net": returned - sent,
        "investor_payoff": cfg.endowment - sent + returned,
        "trustee_payoff": cfg.endowment + tripled - returned,
        "trustee_received": tripled > 0,
        "betrayal": bool(tripled > 0 and returned < sent),
        "full_betrayal": bool(tripled > 0 and returned == 0),
        "reciprocated": bool(tripled > 0 and returned >= sent),
        "investor_raw": d["inv_raw"], "trustee_raw": d["tru_raw"],
        "investor_reasoning": inv_reason, "trustee_reasoning": tru_reason,
        "investor_parse_clean": inv_clean, "trustee_parse_clean": tru_clean,
        "transcript": d["transcript"] or None,
        "batch": True,
        "input_tokens": sum(u["input_tokens"] or 0 for u in usage),
        "output_tokens": sum(u["output_tokens"] or 0 for u in usage),
        "reasoning_tokens": sum(u["reasoning_tokens"] or 0 for u in usage) or None,
        "thinking_blocks": sum(u["thinking_blocks"] or 0 for u in usage) or None,
    }


# --------------------------------------------------------------------------- #
# Per-arm driver                                                              #
# --------------------------------------------------------------------------- #
def alive(pid: int) -> bool:
    """Is that process still running? A stale lock from a crash must not block."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True              # exists but belongs to someone else
    return True


class Arm:
    def __init__(self, key: str, dyads: int):
        self.key = key
        self.spec = P.PACK[key]
        self.dir = ROOT / key
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock()
        self.state_p = self.dir / "state.json"
        self.ledger_p = self.dir / "batches.jsonl"
        self.out_p = self.dir / f"full_{key}.jsonl"
        self.att = ATT.AttritionLog()
        self.att_p = self.dir / f"full_{key}_attrition.jsonl"
        if self.state_p.exists():
            self.state = json.loads(self.state_p.read_text())
        else:
            self.state = plan(dyads)
            self.save()

    def _lock(self) -> None:
        """Take the arm, or refuse to start if a live process already holds it."""
        p = self.dir / "RUNNING.lock"
        if p.exists():
            try:
                held = json.loads(p.read_text())
            except Exception:                                   # noqa: BLE001
                held = {}
            pid = held.get("pid")
            if pid and alive(int(pid)):
                sys.exit(
                    f"\n{self.key} is already being run by pid {pid}, started "
                    f"{held.get('started')}.\n"
                    f"Two runners on one arm submit the same wave twice and the "
                    f"account pays twice.\n"
                    f"Wait for it, or stop it with  kill {pid}  and rerun.\n"
                    f"If you are certain it is gone, delete {p}")
            print(f"  clearing a stale lock from pid {pid}")
        p.write_text(json.dumps({"pid": os.getpid(),
                                 "started": time.strftime("%Y-%m-%d %H:%M:%S")}))
        self.lock_p = p
        atexit.register(self._unlock)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._on_signal)
            except ValueError:                                  # not main thread
                pass

    def _unlock(self) -> None:
        try:
            if self.lock_p.exists():
                held = json.loads(self.lock_p.read_text())
                if held.get("pid") == os.getpid():
                    self.lock_p.unlink()
        except Exception:                                       # noqa: BLE001
            pass

    def _on_signal(self, signum, frame):                        # noqa: ARG002
        self._unlock()
        sys.exit(f"\nstopped by signal {signum}. rerunning resumes.")

    def save(self) -> None:
        tmp = self.state_p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.state_p)              # atomic, so a crash cannot truncate

    def ledger(self) -> list[dict]:
        if not self.ledger_p.exists():
            return []
        return [json.loads(l) for l in self.ledger_p.read_text().splitlines()
                if l.strip()]

    def record_batch(self, wave: str, bid: str, n: int, attempt: int) -> None:
        with self.ledger_p.open("a") as fh:
            fh.write(json.dumps({
                "wave": wave, "batch_id": bid, "requests": n,
                "attempt": attempt, "state": "submitted",
                "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }) + "\n")

    def close_batch(self, bid: str) -> None:
        rows = self.ledger()
        for r in rows:
            if r["batch_id"] == bid:
                r["state"] = "collected"
        self.ledger_p.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def live_batch(self, wave: str) -> Optional[dict]:
        """A submitted but not yet collected batch for this wave, if any."""
        for r in self.ledger():
            if r["wave"] == wave and r["state"] == "submitted":
                return r
        return None


def run_wave(arm: Arm, w, be, dry: bool, poll: int) -> None:
    name = wave_name(w)
    due = [d for d in arm.state if needs(d, w)]

    live = arm.live_batch(name)
    if live is None:
        if not due:
            return
        built = [build_request(d, w, be) for d in due]
        # custom_id -> dyad_id, so the mapping back never depends on the
        # substitution being invertible.
        back = {safe_id(cid): cid for cid, _ in built}
        reqs = [(safe_id(cid), body) for cid, body in built]
        if dry:
            chars = sum(len(json.dumps(b)) for _, b in reqs)
            print(f"  {name:<18}{len(reqs):>6} requests"
                  f"{chars/1e3:>9.0f} kB   (dry run, not submitted)")
            fake = {"ok": True, "text": '{"reasoning": "dry", "amount": 4}',
                    "input_tokens": 0, "output_tokens": 0,
                    "reasoning_tokens": None, "thinking_blocks": None,
                    "stop": "stop"}
            for d in due:
                apply_result(d, w, fake)
            return
        attempt = 1 + sum(1 for r in arm.ledger() if r["wave"] == name)
        tmp = arm.dir / f"{name}_attempt{attempt}.jsonl"
        print(f"  {name:<18}{len(reqs):>6} requests   submitting", flush=True)
        bid = be.submit(reqs, tmp)
        # The id goes to disk BEFORE the first poll. A crash here still leaves a
        # run that can re-attach instead of paying for the same wave twice.
        arm.record_batch(name, bid, len(reqs), attempt)
        print(f"  {'':<18}{'':>6}   batch {bid}")
        log(arm.dir, f"{name} submitted {len(reqs)} requests as {bid}")
    else:
        bid = live["batch_id"]
        back = {safe_id(d["dyad_id"]): d["dyad_id"] for d in due}
        print(f"  {name:<18}{live['requests']:>6} requests   re-attached to {bid}")

    waited = 0
    while True:
        st = be.status(bid)
        if st == "ended":
            break
        try:
            note = be.detail(bid)
        except Exception:                                       # noqa: BLE001
            note = st
        msg = f"{name} {note}, waited {waited//60}m"
        print(f"  {'':<18}{'':>6}   {msg}", flush=True)
        log(arm.dir, msg)
        time.sleep(poll)
        waited += poll

    raw = be.results(bid)
    res = {back.get(k, k): v for k, v in raw.items()}
    ok = bad = 0
    for d in arm.state:
        r = res.get(d["dyad_id"])
        if r is None or not needs(d, w):
            continue
        if r["ok"]:
            apply_result(d, w, r)
            ok += 1
        else:
            bad += 1
            arm.att.add(cell=f"{arm.key}/{d['surface']}/{d['condition']}",
                        dyad_id=d["dyad_id"], role=w[2],
                        cause="infrastructure",
                        snippet=f"batch {name}: {r.get('error')}")
            # An unbilled failure leaves the dyad where it was, so the next pass
            # picks it up. Nothing is lost and nothing was paid for.
    arm.close_batch(bid)
    arm.save()
    arm.att.write(arm.att_p)      # flushed each wave, so a crash keeps the ledger
    print(f"  {'':<18}{'':>6}   {ok} applied, {bad} to retry")
    log(arm.dir, f"{name} collected: {ok} applied, {bad} to retry")


def run_arm(key: str, dyads: int, dry: bool, poll: int, max_retries: int) -> None:
    arm = Arm(key, dyads)
    print(f"\n{key}  ({arm.spec.model}, {arm.spec.extra_body})")
    be = None if dry else backend(arm.spec)
    if dry:
        class _Dry:
            def body(self, s, u):
                return {"system_chars": len(s), "user_chars": len(u)}
        be = _Dry()

    for w in WAVES:
        for attempt in range(max_retries):
            run_wave(arm, w, be, dry, poll)
            left = len([d for d in arm.state if needs(d, w)])
            if not left:
                break
            print(f"  {'':<18}{'':>6}   {left} left, resubmitting")
        stuck = [d for d in arm.state if needs(d, w)]
        for d in stuck:                          # give up on these, log and move on
            d["failed"] = f"{wave_name(w)} exhausted retries"
        if stuck:
            print(f"  {'':<18}{'':>6}   {len(stuck)} abandoned after "
                  f"{max_retries} attempts")
        arm.save()

    done = [d for d in arm.state if d["stage"] == len(WAVES) and not d["failed"]]
    if not dry:
        with arm.out_p.open("w") as fh:
            for d in done:
                fh.write(json.dumps(finish(d, arm.spec), ensure_ascii=False) + "\n")
        print(f"  wrote {len(done)} dyads to {arm.out_p.name}")


def show_status() -> None:
    if not ROOT.exists():
        print("nothing started")
        return
    for d in sorted(ROOT.iterdir()):
        led = d / "batches.jsonl"
        st = d / "state.json"
        if not st.exists():
            continue
        state = json.loads(st.read_text())
        stages = {}
        for x in state:
            stages[x["stage"]] = stages.get(x["stage"], 0) + 1
        live = [json.loads(l) for l in led.read_text().splitlines()
                if l.strip() and json.loads(l)["state"] == "submitted"] if led.exists() else []
        print(f"\n{d.name}")
        print(f"  dyads by stage: " +
              ", ".join(f"{k}:{v}" for k, v in sorted(stages.items())))
        print(f"  complete: {stages.get(len(WAVES), 0)} of {len(state)}")
        for b in live:
            print(f"  IN FLIGHT  {b['wave']:<16}{b['batch_id']}")
            print(f"             {b['requests']} requests, submitted "
                  f"{b['submitted_at']}")
            try:
                be = backend(P.PACK[d.name])
                print(f"             provider says: {be.detail(b['batch_id'])}")
            except SystemExit as exc:
                # The backend raises SystemExit when a key is absent, and
                # SystemExit is not an Exception, so catching only Exception let
                # it kill the whole status view. A missing key is a reason to
                # skip the remote question, not to stop reporting local state.
                print(f"             provider not queried: {exc}")
            except Exception as exc:                            # noqa: BLE001
                print(f"             could not reach the provider: "
                      f"{type(exc).__name__}: {exc}")
        if not live:
            print("  nothing in flight")
        plog = d / "progress.log"
        if plog.exists():
            tail = plog.read_text().splitlines()[-3:]
            for line in tail:
                print(f"  log  {line}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="all")
    ap.add_argument("--dyads", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--poll", type=int, default=POLL_SECONDS)
    ap.add_argument("--max-retries", type=int, default=3)
    a = ap.parse_args()

    if a.status:
        show_status()
        return

    arms = ARMS_ALL if a.arms == "all" else [x.strip() for x in a.arms.split(",")]
    unknown = [x for x in arms if x not in P.PACK]
    if unknown:
        sys.exit(f"unknown arm: {', '.join(unknown)}")

    for key in arms:
        try:
            run_arm(key, a.dyads, a.dry_run, a.poll, a.max_retries)
        except KeyboardInterrupt:
            print("\ninterrupted. any live batch is in batches.jsonl and "
                  "--resume re-attaches to it.")
            raise
        except Exception as exc:                                # noqa: BLE001
            print(f"  {key} STOPPED: {type(exc).__name__}: {exc}")
            print("  the ledger keeps the batch id, so rerunning re-attaches "
                  "instead of resubmitting")


if __name__ == "__main__":
    main()
