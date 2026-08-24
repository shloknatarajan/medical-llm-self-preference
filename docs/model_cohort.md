# Expanded Model Cohort

This document defines the primary model cohort for the medical LLM
self-preference experiments. The cohort was selected on August 23, 2026.

The study uses two strong capability tiers from four model families. Small and
efficiency-oriented models are excluded because insufficient generation quality
or judging ability could obscure the self-preference effect. Stable endpoints
are preferred to reduce the risk of endpoint changes or deprecation during the
experiment. Gemini 3.1 Pro Preview is the sole exception because Google does not
currently offer a stable Gemini 3-series Pro endpoint, and using Gemini 2.5 Pro
would introduce a substantial model-vintage mismatch with the other frontier
models in the cohort.

## Primary cohort

| Family | Higher-capability model | Strong workhorse model |
|---|---|---|
| OpenAI | `gpt-5.6-sol` | `gpt-5.6-terra` |
| Anthropic | `claude-opus-5` | `claude-sonnet-5` |
| Google | `gemini-3.1-pro-preview` | `gemini-3.7-flash` |
| Qwen | `Qwen/Qwen3.5-122B-A10B` | `Qwen/Qwen3.8-27B` |

All eight models will be used as both response generators and judges.

The tier labels describe the models' intended roles in this study rather than
a universal performance ranking. The higher-capability Qwen tier uses the
122B-total/10B-active mixture-of-experts model instead of Qwen3.8-2.4T-A95B so
that both Qwen tiers can be deployed practically on Modal. The Qwen models are
from different generations, which should be recorded as a limitation in
within-family analyses.

## Baseline patient simulator

The baseline standardized-patient model for the MedSP1000 multi-turn
generation is `mistralai/Mistral-Small-3.1-24B-Instruct-2503`. This model is a
simulation component, not an additional response generator or judge in the
primary eight-model cohort.

Mistral Small 3.1 is the baseline because the patient is grounded in a private,
role-specific MedSP1000 actor packet and primarily needs reliable instruction
following and natural role-play rather than independent medical reasoning. In
the earlier 50-case paired patient-simulator pilot, it completed all 150 turns
without an empty reply and was faster than `Qwen/Qwen3.8-27B-FP8`. Because it
tended to produce longer replies, the MedSP1000 prompt should require concise
answers and incremental disclosure of packet facts. A small MedSP1000 smoke set
should be reviewed for realism before generating the full cohort.

`Qwen/Qwen3.8-27B-FP8` remains the higher-capability patient-simulator
comparison model, rather than the baseline.

## Selection rules

- Prefer stable or generally available model endpoints in the primary analysis.
- Use the exact `gemini-3.1-pro-preview` endpoint for the higher-capability
  Google tier. Record its preview status as a study limitation and complete its
  generation and judging runs within a bounded collection window.
- Do not use provider aliases such as `latest` that can silently change during
  data collection.
- Pin the exact repository commit, weight format, quantization, inference
  engine, and inference-engine version for every open-weight model.
- Record the provider, exact model identifier, API version, access date,
  reasoning configuration, output-token limit, and generation parameters with
  every run.
- Disable external tools and retrieval for both generation and judging.

## Medical competency gate

Before the full experiment, each proposed judge should pass the same held-out
medical evaluation. The gate should measure:

- agreement with expert scores and pairwise preferences;
- recognition of safety-critical errors;
- rubric and structured-output compliance;
- test-retest consistency; and
- completion reliability for eight-turn conversations.

A model that fails the preregistered judging threshold may remain in the
generator cohort, but it should not be treated as a primary judge.

## Scale

With both generators judging every pair, 125 scenarios, one single-turn
condition, and four multi-turn lengths, this 8-model cohort produces 35,000
pairwise judgments before sibling-family calibration judgments, expert review,
retries, or malformed-output reruns.

For `S` scenarios, the corresponding number is:

```text
8 * 7 * S * 5 = 280S judgments
```

## Model references

- [OpenAI models](https://developers.openai.com/api/docs/models)
- [Anthropic models](https://platform.claude.com/docs/en/about-claude/models/overview)
- [Gemini models](https://ai.google.dev/gemini-api/docs/models)
- [Qwen models](https://huggingface.co/Qwen/models)
