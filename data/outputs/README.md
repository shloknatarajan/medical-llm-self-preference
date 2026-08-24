# Generation outputs

Generation outputs are append-only JSONL files. Each row is one generation
attempt and uses the same outer envelope across single-turn and multi-turn
experiments:

- `experiment_id`, `run_id`, `generation_key`, and `generation_id`;
- `attempt`, `status`, and `created_at`;
- `question_id` and the frozen question inputs;
- exact model and prompt identifiers;
- generation parameters, token usage, latency, and finish state; and
- nullable error fields.

## MedSP1000 multi-turn responses

MedSP1000 uses
[`schemas/medsp1000_multiturn_generation.schema.json`](schemas/medsp1000_multiturn_generation.schema.json).
One row stores one complete clinician-patient conversation. Messages are not
split across JSONL rows, because a partial join could silently mix attempts or
runs.

The ordered `turns` array is the canonical response. Every turn records:

```json
{
  "turn_index": 1,
  "exchange_index": 1,
  "role": "clinician",
  "content": "Hello, what brings you in today?",
  "model": "Qwen/Qwen3.5-122B-A10B-FP8",
  "finish_reason": "stop",
  "input_tokens": 314,
  "output_tokens": 11,
  "latency_ms": 920
}
```

`turn_index` orders all messages from 1 through 8. `exchange_index` groups each
clinician turn with the patient reply it elicited. `transcript_text` is a
derived human-readable rendering; analysis should use `turns`.

The row repeats the full `question_text` and `private_patient_context`, matching
the single-turn convention of preserving the exact input next to its response.
Their hashes link the response back to the frozen question set and detect input
drift.

Each invocation also writes `<run_id>.manifest.json` beside the JSONL. The
manifest stores the question IDs, input artifact hash, model pair, full prompt
templates (including current-turn controls) and hashes, inference configuration,
checkpoint batch size, run status, and live completion counts. Each completed
conversation row is appended and synced individually; partial conversations are
never written. A `generation_key` includes the prompt version, exchange count,
output caps, sampling configuration, and seed in addition to the question and
model pair, so resume behavior cannot conflate materially different inputs or
generation settings.
