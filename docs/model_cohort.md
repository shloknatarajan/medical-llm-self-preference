# Expanded Model Cohort

This document defines the primary model cohort for the medical LLM
self-preference experiments. The cohort was selected on August 23, 2026.

The study uses two strong capability tiers from four model families. Small and
efficiency-oriented models are excluded because insufficient generation quality
or judging ability could obscure the self-preference effect. Preview models are
also excluded from the primary analysis to reduce the risk of endpoint changes
or deprecation during the experiment.

## Primary cohort

| Family | Higher-capability model | Strong workhorse model |
|---|---|---|
| OpenAI | `gpt-5.6-sol` | `gpt-5.6-terra` |
| Anthropic | `claude-opus-5` | `claude-sonnet-5` |
| Google | `gemini-2.5-pro` | `gemini-3.7-flash` |
| Qwen | `Qwen/Qwen3.5-122B-A10B` | `Qwen/Qwen3.8-27B` |

All eight models will be used as both response generators and judges.

The tier labels describe the models' intended roles in this study rather than
a universal performance ranking. The higher-capability Qwen tier uses the
122B-total/10B-active mixture-of-experts model instead of Qwen3.8-2.4T-A95B so
that both Qwen tiers can be deployed practically on Modal. The Qwen models are
from different generations, which should be recorded as a limitation in
within-family analyses.

## Selection rules

- Use stable or generally available model endpoints in the primary analysis.
- Do not use `gemini-3.1-pro-preview`; use stable `gemini-2.5-pro` instead.
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
