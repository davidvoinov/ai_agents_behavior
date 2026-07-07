# Model Matrix

This file tracks which model families and providers are planned for pilot and full data collection.

The study should compare model families rather than treating providers as the main unit of analysis. A provider is the API platform used to access a model. A model family is the scientific category used in analysis.

## Pilot Models

The pilot should prioritize models that are free, already accessible, or cheap enough to use during debugging.

| Status | Model family | Provider | Example model ID | Notes |
|---|---|---|---|---|
| Baseline | GPT-OSS | Groq | `openai/gpt-oss-20b` | Already represented in existing runs. |
| Baseline | GPT-OSS | Cerebras | `gpt-oss-120b` | Already represented in existing runs. |
| Pilot | Llama | Groq or OpenRouter | Provider-dependent | Open-weight comparison model. |
| Pilot | Qwen | Groq or OpenRouter | Provider-dependent | Open-weight comparison model. |
| Pilot | Gemini | Google Gemini API | Provider-dependent | Useful free-tier model family. |
| Pilot | DeepSeek | OpenRouter, Groq, or direct API | Provider-dependent | Include if accessible at low cost. |

## Full-Study Models

The full study should include both proprietary and open-weight model families.

| Status | Model family | Provider | Notes |
|---|---|---|
| Full study | OpenAI proprietary GPT | OpenAI API | Necessary commercial baseline distinct from GPT-OSS. |
| Full study | Claude | Anthropic API | Important non-OpenAI proprietary family. |
| Full study | Gemini | Google Gemini API | Important Google model family. |
| Full study | Mistral | Mistral API or OpenRouter | European model family. |
| Full study | DeepSeek | DeepSeek API or OpenRouter | Low-cost reasoning-oriented family. |
| Full study | Llama | Groq, OpenRouter, Together, or other provider | Open-weight baseline. |
| Full study | Qwen | Groq, OpenRouter, Together, or other provider | Open-weight baseline. |

## Pairing Strategy

The first stage should use same-model pairs to estimate each model family's baseline behavior.

Examples:

- GPT-OSS vs GPT-OSS;
- Gemini vs Gemini;
- Llama vs Llama;
- Qwen vs Qwen.

The second stage should use cross-model pairs. For each cross-model pair, roles should be reversed.

Examples:

- Gemini as investor, Qwen as trustee;
- Qwen as investor, Gemini as trustee;
- Llama as investor, DeepSeek as trustee;
- DeepSeek as investor, Llama as trustee.

## Pricing and Access Notes

Pricing and free-tier availability should be checked immediately before data collection. The checked date should be recorded in the final experiment log.

For analysis, model metadata should include:

- provider;
- model ID;
- model family;
- access status;
- pricing snapshot date;
- whether the run used a free tier, paid tier, or institutional account.
