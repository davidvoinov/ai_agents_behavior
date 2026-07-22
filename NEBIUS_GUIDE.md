# Nebius Token Factory — practical guide for the Berg runs

Nebius Token Factory is the pilot workhorse. It serves open-weight models
(DeepSeek, Qwen, Llama, Mistral, GPT-OSS and others) over an OpenAI-compatible
API with **autoscaling and no rate throttling**, which is exactly what removes
the rate-limit dropouts that produced the attrition in Table 3. Closed frontier
models (GPT-5.6, Claude) are not on Nebius — reach those through their own API
plus Batch.

## 1. Account and key

1. Sign up at <https://studio.nebius.com> (Nebius AI Studio / Token Factory).
   New accounts get a starting credit.
2. Create an API key in the console.
3. Put it in the project-root `.env` (the same file the runners already read):

   ```
   NEBIUS_API_KEY=...
   DEEPSEEK_API_KEY=...        # DeepSeek direct — 5M free tokens on signup
   MOONSHOT_API_KEY=...
   MINIMAX_API_KEY=...
   ZHIPU_API_KEY=...
   OPENROUTER_API_KEY=...      # serves Claude/GPT OpenAI-compatibly
   OPENAI_API_KEY=...
   ```

## 2. Endpoint and model IDs

- OpenAI-compatible base URL: `https://api.studio.nebius.com/v1/`
  (confirm the exact host in your console — Nebius has used both
  `api.studio.nebius.com` and `api.studio.nebius.ai`).
- Model IDs are namespaced, e.g. `deepseek-ai/DeepSeek-V3`,
  `Qwen/Qwen3-235B-A22B`, `meta-llama/Meta-Llama-3.1-70B-Instruct`.
  **Copy the exact IDs from the Models page** — they change, and
  `berg/providers.py` holds placeholders you must reconcile once.

Everything is OpenAI-compatible, so the existing `make_chat_fn` works unchanged.
`berg/providers.py` already points the `nebius` provider at this base URL.

## 3. Using it from the harness

```python
from providers import model_config, chat_for

# open model via Nebius (no throttling)
chat = chat_for("qwen3-think")                 # provider defaults to nebius
chat_plain = chat_for("qwen3-nothink")         # same weights, thinking OFF

# provider-confound control (PI point 4): SAME model, second provider
chat_alt = chat_for("qwen3-think", provider="openrouter")
```

The reasoning on/off toggle for Qwen3 is wired through `extra_body`
(`{"chat_template_kwargs": {"enable_thinking": false}}`); confirm Nebius honors
this flag for the specific Qwen build, otherwise fall back to appending
`/no_think` to the system prompt.

## 4. Combining DeepSeek free tokens with Nebius

They are two independent free pools — use both, do not choose.

- **DeepSeek direct** gives ~5M free tokens on signup, applied automatically to
  all DeepSeek models. Spend them on the DeepSeek pilot (V4 Pro and V4 Flash,
  the reasoning toggle) by keeping `provider="deepseek"` for those two keys.
  Note the direct API deprecates `deepseek-chat`/`deepseek-reasoner` on
  2026-07-24 — use `deepseek-v4-flash` / `deepseek-v4-pro`.
- **Nebius** starting credit covers every other open model with no throttling,
  and also hosts DeepSeek once the 5M free tokens run out (then just pass
  `provider="nebius"` to the DeepSeek keys).

Practical split for the pilot:

```python
# DeepSeek on its own free credit
ds_pro   = chat_for("deepseek-v4-pro",   provider="deepseek")
ds_flash = chat_for("deepseek-v4-flash", provider="deepseek")
# everything else on Nebius, no throttling
qwen     = chat_for("qwen3-think",  provider="nebius")
glm      = chat_for("glm-4.6",      provider="nebius")   # if hosted; else provider="zhipu"
```

The keys live in `.env` as `DEEPSEEK_API_KEY` and `NEBIUS_API_KEY`; the runner
only swaps `base_url` and the key, nothing else.

## 5. Throughput and pacing

Nebius autoscales with no rate throttling, so you can drop the
`MIN_CALL_INTERVAL` pacing that the Groq and Cerebras free tiers forced. Run
concurrent requests to finish the confirmatory set quickly. Keep `with_retry`
in place anyway — a transient 5xx is still logged as an infrastructure dropout
by `attrition.py`, which is what keeps the attrition split honest.

## 7. Launch checklist (recognition pilot on Nebius)

1. **Key.** Create an API key at <https://studio.nebius.com>, add `NEBIUS_API_KEY=...`
   to the project-root `.env`.
2. **Deps.** `pip install -r requirements.txt` (langchain-openai, langchain-core,
   python-dotenv).
3. **Confirm model IDs.** Open the Nebius Models page (or the
   [token-factory-cookbook](https://github.com/nebius/token-factory-cookbook))
   and copy the exact IDs, then paste them into `berg/providers.py` (`model=`
   field of each `nebius` spec). The placeholders to verify: `qwen3-think` /
   `qwen3-nothink`, `glm-4.6`, `minimax-m2`, `kimi-k2.6`. Nebius also carries
   `DeepSeek V4 Pro` and `gpt-oss-120b`, so the reasoning models can run here
   with no throttling — add them as nebius specs if you want to avoid the
   Cerebras 5 req/min cap.
4. **Smoke (2 dyads, 1 surface).**
   ```
   cd berg
   MODELS="qwen3-think" SURFACES="canonical" PILOT_N=2 python run_pilot_berg.py
   ```
   Expect a `smoke: {...ok...}` line, one `mean_sent` line, and an attrition
   summary with `infra=0`.
5. **Recommended pilot — the reasoning toggle (tests H2/H3 surface effects × H4
   reasoning), no throttling:**
   ```
   MODELS="qwen3-think,qwen3-nothink" \
   SURFACES="canonical,reskin,shifted,reskin_shifted,strong_novel" \
   PILOT_N=20 python run_pilot_berg.py
   ```
6. **Power.** `python power_analysis.py data/pilot/pilot_<ts>.jsonl --contrast canonical strong_novel`
7. **No pacing needed** on Nebius (autoscaling). Keep `with_retry`; a transient
   5xx is logged as infrastructure by `attrition.py`, never mixed with behavior.
8. **If `enable_thinking` is ignored** by the Nebius Qwen build (the no-think arm
   looks identical to think), fall back to appending `/no_think` to the system
   prompt (Qwen soft switch) — tell me and I will wire it.

Cost: the toggle pilot above (2 models × 5 surfaces × 20 = 200 no-history dyads,
Qwen3 at ~$0.20/$0.60) is about **$0.25** off the Nebius credit.

## 6. What still needs the direct API

- **Claude Sonnet 5, GPT-5.6** are not on Nebius. Use Anthropic and OpenAI
  directly (or OpenRouter, which is OpenAI-compatible), and use their **Batch
  API** for the confirmatory pass: 50% cheaper and under separate, high limits
  that avoid rate-limit dropouts. Batch fits the independent cells
  (no_history, social_history, promise-switch, and the judge pass); the
  communication cells run their message turns sequentially within each dyad.
