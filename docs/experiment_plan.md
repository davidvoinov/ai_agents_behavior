# Multi-Model Trust Game Study

## Research Question

Will two large language models, playing a sequential trust game with each other, form a cooperative agreement aimed at maximizing joint payoff and sharing the surplus, despite the Nash equilibrium prediction of no trust and no return?

The core empirical object is not simply cooperation, but the full sequence:

1. whether an agreement is explicitly formed;
2. whether the agreement prescribes a cooperative strategy;
3. whether the final private choices follow the agreement;
4. whether the observed behavior differs from the no-communication baseline.

## Hypotheses

**H1. Communication and explicit agreement**

When pre-play communication is available, LLM pairs will explicitly agree on a cooperative strategy. In the Berg trust game this means that the investor sends the full endowment, or a large share of it, and the trustee promises to return a positive amount that leaves the investor better off than keeping the initial endowment.

**H2. Agreement execution**

Agreements reached during communication will be executed more often than cooperation appears in human no-communication trust-game experiments. The key outcome is not only whether trust occurs, but whether the trustee honors the specific promise made during communication.

**H3. No-communication behavior**

Without communication, LLMs will either reproduce the qualitative pattern found in human trust-game experiments, namely nonzero trust and reciprocity, or display behavior closer to the Nash equilibrium prediction, namely zero transfer and zero return.

## Existing Baseline

The current repository already contains an initial GPT-OSS baseline. The existing runs use OpenAI OSS models through external inference providers, mainly Groq and Cerebras. These runs are useful as a technical baseline, but they do not yet answer the broader multi-model question because they mostly cover one model family.

The next stage should therefore separate three concepts that are currently easy to conflate:

- **provider**: the API or inference platform used to call the model;
- **model_id**: the concrete model name exposed by the provider;
- **model_family**: the broader model lineage used for scientific comparison.

The main study should compare model families, not only providers.

## Core Experiments

### Experiment 1: Berg Trust Game Without Communication

This is the baseline experiment for H3.

The investor receives 10 units and chooses how many units to send to the trustee. The sent amount is tripled. The trustee then chooses how much of the tripled amount to return to the investor. No communication, social history, or information about previous runs is provided.

This condition estimates the baseline level of trust and reciprocity for each model pair.

### Experiment 2: Berg Trust Game With Pre-Play Communication

This is the main experiment for H1 and H2.

Before the final choices are made, the investor and the trustee exchange a limited number of messages. After communication ends, both players make private final decisions. The communication transcript is coded for agreement formation, cooperative content, promised transfers, and agreement execution.

This condition tests whether LLMs explicitly coordinate on joint-payoff maximization and whether such coordination survives the final private decision stage.

### Experiment 3: Berg Trust Game With Human Social History

This condition follows the logic of the social-history treatment in Berg, Dickhaut, and McCabe (1995). Models are shown a short summary of prior human behavior from the original experiment before making their choices.

The goal is to test whether LLMs respond to human social evidence and whether human precedent increases trust or reciprocity.

### Experiment 4: Berg Trust Game With LLM Social History

Models are shown a short summary of previous LLM behavior from earlier runs. This condition is distinct from human social history and should be labeled separately.

The goal is to test whether models react differently to human social norms and to the observed behavior of other models.

### Experiment 5: Same-Model and Cross-Model Pairing

The main experiments should be run in both same-model and cross-model pairings.

Same-model examples:

- GPT-OSS vs GPT-OSS;
- Gemini vs Gemini;
- Llama vs Llama;
- Qwen vs Qwen.

Cross-model examples:

- GPT-OSS vs Gemini;
- Llama vs Qwen;
- Gemini vs Qwen;
- Llama vs DeepSeek.

For cross-model pairs, each model should appear in both roles. For example, Gemini should play as investor against Qwen as trustee, and Qwen should also play as investor against Gemini as trustee.

## Main Outcomes

For all Berg trust-game conditions:

- amount sent by the investor;
- amount returned by the trustee;
- investor payoff;
- trustee payoff;
- whether the investor sent zero;
- whether the investor sent the full endowment;
- whether the trustee returned zero after receiving a positive transfer;
- whether the trustee returned more than the investor sent;
- whether the final allocation increased joint surplus.

For communication conditions:

- whether an explicit agreement was formed;
- whether the agreement specified the investor's transfer;
- whether the agreement specified the trustee's return;
- whether the agreement involved full or near-full investment;
- whether the promised return made the investor better off than not investing;
- whether the investor followed the agreement;
- whether the trustee followed the agreement;
- whether either player deviated from the agreement.

## Pilot Stage

The first pilot should use only free or already accessible models. The purpose of the pilot is not to make final statistical claims. Its purpose is to check that prompts, parsing, logging, payoff calculation, and model-pair configuration work reliably.

The pilot size should be 10 dyads per condition and model pair.

The first pilot should focus on Experiment 1. Once the baseline runner is reliable, the same structure can be extended to communication and social-history treatments.

## Full Data Collection

The full study should be run only after the pilot confirms that:

- all prompts are clear;
- model responses are consistently parseable;
- invalid choices are detected and logged;
- raw responses are preserved;
- model family and provider are recorded separately;
- same-model and cross-model pairings are correctly balanced.

Paid models should be added only after this stage, so that paid calls are used for final data collection rather than debugging.

## Analysis Plan

The first analysis should compare behavior across treatments and model families using:

- descriptive statistics by model pair and treatment;
- comparison of mean sent and mean returned amounts;
- comparison of zero-transfer rates;
- comparison of reciprocity rates;
- comparison of agreement formation and agreement execution rates;
- role-specific behavior in cross-model pairings.

The final analysis should test whether communication increases cooperation relative to the no-communication baseline and whether this effect differs across model families.
