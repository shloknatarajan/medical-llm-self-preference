# Generation pipelines

## MedSP1000 multi-turn generation

Build the frozen 200-question set directly from the pinned MedSP1000 source:

```bash
uv run python scripts/prepare_medsp1000_questions.py
```

Run one end-to-end smoke question or the full set. Mistral runs on Modal; the
selected clinician is routed to Modal or its provider API:

```bash
uv run modal run src/generation/modal_medsp1000.py --smoke-test

uv run modal run src/generation/modal_medsp1000.py \
  --clinician-model claude-sonnet-5 \
  --num-questions 200
```

The runner generates 8 conversations at a time by default. As soon as a batch
finishes, each complete conversation is appended and synced individually, then
the run manifest is updated with progress. If a run is interrupted, the next
invocation skips every durably saved conversation. Adjust the tradeoff between
inference batching and potential rework with
`--checkpoint-batch-size` (use `1` for per-conversation checkpoints).
Provider, truncation, and batch-validation failures are also appended as failed
attempt rows, including any turns completed before the failure. They remain
eligible for a later retry and advance the attempt counter on the next run.

The patient simulator is fixed to
`mistralai/Mistral-Small-3.1-24B-Instruct-2503` on Modal. The clinician is an
experimental variable selected with `--clinician-model`; OpenAI, Anthropic,
Gemini, and the two pinned Qwen/Modal models use the same conversation and
checkpoint pipeline. Role contexts remain separate throughout generation.
Prior visible messages are kept clean; patient grounding and clinician turn
controls are attached only to the current message. Embedded role context hashes
are verified before inference. Outputs are append-only and resumable by
question, model pair, prompt version, exchange count, output caps, sampling
configuration, and seed at
`data/outputs/medsp1000/generations.jsonl`; each run writes a separate manifest
beside it. One JSONL row contains one complete conversation attempt, with
turn-level model, token, latency, and finish metadata.
Both question and output rows are validated against their committed JSON
Schemas. Output loading additionally verifies turn ordering, transcript
derivation, and token/latency aggregates before resume state is accepted.
API clinicians use an explicitly pinned medium reasoning effort. The
4,096-token clinician generation budget leaves room for hidden reasoning; the
prompt itself asks for concise natural messages without imposing a
response-length scoring target. Qwen reasoning remains disabled until a
comparable open-weight reasoning condition is selected for that cohort.

## Real-POCQi single-turn generation

The Real-POCQi pipeline deterministically shuffles the committed 620-question
artifact with seed 42 and asks each requested model to answer as the question's
specialty expert. Use the question-count option only for an intentional sample.

```bash
uv run python -m generation.generate_real_pocqi \
  --models gpt-5.6-sol gpt-5.6-terra claude-opus-5 claude-sonnet-5
```

The CLI loads provider credentials from the repository's ignored `.env` file
without replacing variables already exported in the shell. Sampling temperature
is omitted by default because the primary OpenAI and Anthropic cohort endpoints
do not accept it. Use `--temperature` only with a compatible model. The default
output cap is 4,096 tokens; responses that still hit the cap are recorded as
failures instead of being mistaken for complete answers.

Mixed-model runs queue models round-robin for each question, allowing providers
to generate concurrently within the configured worker limit.

Qwen and other Modal-hosted models also require their deployed app:

```bash
uv run python -m generation.generate_real_pocqi \
  --models modal/Qwen3.6-35B \
  --modal-app medical-llm-inference
```

The pinned Qwen cohort has a dedicated batched Modal runner. It serves the
official FP8 checkpoints for Qwen3.8-27B on one H100 and
Qwen3.5-122B-A10B on two H100s, with thinking disabled:

```bash
uv run modal run src/generation/modal_real_pocqi_generation.py
```

The dedicated runner defaults to the full 620-question cohort and a 2,048-token
output cap. To upgrade previously capped attempts while preserving completed
answers, pass `--regenerate-truncated`; only attempts truncated under a smaller
cap are regenerated.

Attempt records are appended immediately to
`data/outputs/generations/real_pocqi_generations.jsonl`. Each failed retry is
retained. Later invocations skip successful question/model keys by default, so
the command can safely resume an interrupted run. Use `--force` only when a new
attempt is intentionally required.

Run `uv run python -m generation.generate_real_pocqi --help` for sampling,
specialty filtering, concurrency, retry, and output options.
