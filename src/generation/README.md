# Generation pipelines

## Real-POCQi single-turn generation

The Real-POCQi pipeline deterministically shuffles the committed 620-question
artifact with seed 42, selects 125 questions, and asks each requested model to
answer as the question's specialty expert.

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

Qwen and other Modal-hosted models also require their deployed app:

```bash
uv run python -m generation.generate_real_pocqi \
  --models modal/Qwen3.6-35B \
  --modal-app medical-llm-inference
```

Attempt records are appended immediately to
`data/outputs/generations/real_pocqi_generations.jsonl`. Each failed retry is
retained. Later invocations skip successful question/model keys by default, so
the command can safely resume an interrupted run. Use `--force` only when a new
attempt is intentionally required.

Run `uv run python -m generation.generate_real_pocqi --help` for sampling,
specialty filtering, concurrency, retry, and output options.
