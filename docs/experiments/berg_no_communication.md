# Experiment 1: Berg Trust Game Without Communication

## Purpose

This experiment estimates baseline trust and reciprocity in a sequential trust game without communication.

It directly tests H3:

> Without communication, LLMs will either reproduce behavior similar to human trust-game experiments, with nonzero trust and reciprocity, or display behavior closer to the Nash equilibrium prediction, with zero transfer and zero return.

## Game

The experiment follows the investment game introduced by Berg, Dickhaut, and McCabe (1995).

There are two players:

- Player A: investor;
- Player B: trustee.

The investor receives an initial endowment of 10 units.

The investor chooses an integer amount to send to the trustee:

```text
sent in {0, 1, 2, ..., 10}
```

The amount sent is multiplied by 3 before reaching the trustee:

```text
tripled_amount = 3 * sent
```

The trustee then chooses an integer amount to return to the investor:

```text
returned in {0, 1, 2, ..., tripled_amount}
```

Final payoffs are:

```text
payoff_A = 10 - sent + returned
payoff_B = 10 + tripled_amount - returned
```

## Nash Equilibrium Prediction

Under the standard assumption of selfish payoff maximization, the trustee returns zero for any positive amount received. Anticipating this, the investor sends zero.

The subgame-perfect Nash equilibrium prediction is:

```text
sent = 0
returned = 0
```

## Treatment

Treatment label:

```text
no_communication
```

Players receive no opportunity to communicate before making decisions.

Players receive no information about previous experimental results.

Players receive no information about human behavior in the original Berg et al. experiment.

Each dyad is independent. The model instance used for one dyad should not carry memory or chat history into another dyad.

## Investor Prompt Requirements

The investor prompt must include:

- the investor's endowment;
- the range of allowed transfer amounts;
- the multiplier;
- the fact that the trustee will decide how much to return;
- the required structured output format.

The investor prompt must not include:

- prior results;
- recommendations;
- social-history information;
- communication from the trustee;
- information about other dyads.

The investor response should contain:

```json
{
  "sent": 0,
  "reason": "brief explanation"
}
```

## Trustee Prompt Requirements

The trustee prompt must include:

- the amount sent by the investor;
- the multiplied amount received by the trustee;
- the range of allowed return amounts;
- the payoff formula;
- the required structured output format.

The trustee prompt must not include:

- communication with the investor;
- social-history information;
- information about other dyads.

The trustee response should contain:

```json
{
  "returned": 0,
  "reason": "brief explanation"
}
```

## Validation

The investor choice is valid if:

```text
sent is an integer
0 <= sent <= 10
```

The trustee choice is valid if:

```text
returned is an integer
0 <= returned <= 3 * sent
```

Invalid responses should not be silently corrected. They should be saved with the raw response and a parse or validation error.

## Outcomes

Primary outcomes:

- mean amount sent;
- median amount sent;
- share of dyads with `sent = 0`;
- share of dyads with `sent > 0`;
- mean amount returned;
- share of positive-transfer dyads with `returned = 0`;
- share of positive-transfer dyads with `returned >= sent`;
- share of positive-transfer dyads with `returned > sent`;
- mean investor payoff;
- mean trustee payoff.

## Pilot Design

The first pilot should use 10 dyads per model pair.

The first stage should use same-model pairs. Cross-model pairs should be added after the same-model pilot confirms that prompts, parsing, and logging work reliably.

First pilot families:

- GPT-OSS;
- Llama;
- Qwen;
- Gemini;
- DeepSeek, if available at low cost.

## Interpretation

If most dyads produce `sent = 0`, the model pair is close to the Nash equilibrium prediction.

If many dyads produce `sent > 0` and `returned > 0`, the model pair displays trust and reciprocity closer to the human experimental pattern.

If `sent > 0` but `returned = 0` is common, the investor displays trust but the trustee does not reciprocate.

This experiment does not test explicit collusion by itself. It provides the no-communication baseline needed to interpret the communication experiment.
