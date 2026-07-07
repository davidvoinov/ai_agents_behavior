# Data Schema

This document defines the minimum fields that should be saved for each dyad in the trust-game experiments.

The guiding principle is that every record should be reproducible, auditable, and analyzable without relying on hidden state from the run script.

## Core Identifiers

| Field | Description |
|---|---|
| `run_id` | Unique identifier for a full experimental run. |
| `experiment_id` | Experiment name, for example `berg_no_communication`. |
| `treatment` | Treatment label, for example `no_communication`, `communication`, `human_social_history`, or `llm_social_history`. |
| `dyad_id` | Pair identifier within the run. |
| `timestamp` | Time at which the record was written. |

## Model Information

| Field | Description |
|---|---|
| `provider_A` | API provider used for the investor. |
| `provider_B` | API provider used for the trustee. |
| `model_id_A` | Provider-specific model identifier for the investor. |
| `model_id_B` | Provider-specific model identifier for the trustee. |
| `model_family_A` | Scientific model-family label for the investor. |
| `model_family_B` | Scientific model-family label for the trustee. |
| `role_A` | Always `investor` in the Berg trust game. |
| `role_B` | Always `trustee` in the Berg trust game. |

## Game Choices

| Field | Description |
|---|---|
| `endowment` | Initial endowment of the investor. |
| `multiplier` | Multiplier applied to the amount sent. |
| `sent` | Amount sent by the investor. |
| `tripled_amount` | Amount received by the trustee after multiplication. |
| `returned` | Amount returned by the trustee. |
| `payoff_A` | Final payoff of the investor. |
| `payoff_B` | Final payoff of the trustee. |

## Prompt and Response Logging

| Field | Description |
|---|---|
| `system_prompt_A` | System or role prompt shown to the investor. |
| `decision_prompt_A` | Decision prompt shown to the investor. |
| `raw_response_A` | Raw response returned by the investor model. |
| `parsed_response_A` | Parsed structured response from the investor. |
| `system_prompt_B` | System or role prompt shown to the trustee. |
| `decision_prompt_B` | Decision prompt shown to the trustee. |
| `raw_response_B` | Raw response returned by the trustee model. |
| `parsed_response_B` | Parsed structured response from the trustee. |

## Parsing and Validation

| Field | Description |
|---|---|
| `parse_success_A` | Whether the investor response was parsed successfully. |
| `parse_success_B` | Whether the trustee response was parsed successfully. |
| `validation_success_A` | Whether the investor choice passed game-rule validation. |
| `validation_success_B` | Whether the trustee choice passed game-rule validation. |
| `parse_error_A` | Parsing or validation error for the investor, if any. |
| `parse_error_B` | Parsing or validation error for the trustee, if any. |

## Communication Fields

These fields apply only to communication treatments.

| Field | Description |
|---|---|
| `communication_rounds` | Number of communication rounds allowed. |
| `communication_transcript` | Full pre-play communication transcript. |
| `agreement_detected` | Whether the transcript contains an explicit agreement. |
| `agreed_sent` | Amount the investor agreed to send, if specified. |
| `agreed_returned` | Amount the trustee agreed to return, if specified. |
| `agreement_maximizes_joint_payoff` | Whether the agreement prescribes full investment or otherwise maximizes joint surplus. |
| `agreement_benefits_investor` | Whether the promised return leaves the investor better off than not investing. |
| `investor_followed_agreement` | Whether the investor's final choice matches the agreement. |
| `trustee_followed_agreement` | Whether the trustee's final choice matches the agreement. |
| `agreement_betrayed` | Whether one side deviated after an agreement was reached. |

## Derived Variables

These variables can be computed during analysis, but may also be saved for convenience.

| Field | Description |
|---|---|
| `sent_zero` | Whether `sent == 0`. |
| `sent_full` | Whether `sent == endowment`. |
| `positive_trust` | Whether `sent > 0`. |
| `returned_zero_after_trust` | Whether `sent > 0` and `returned == 0`. |
| `returned_at_least_sent` | Whether `returned >= sent`. |
| `returned_more_than_sent` | Whether `returned > sent`. |
| `joint_payoff` | Sum of investor and trustee payoffs. |
