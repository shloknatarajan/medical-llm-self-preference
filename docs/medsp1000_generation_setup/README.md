# MedSP1000 Multi-Turn Generation Setup

This setup creates realistic, text-only conversations between a fixed
standardized-patient simulator and a clinician model under evaluation. It uses
the role-separated materials in MedSP1000 and intentionally does not perform
grading or judging during generation.

## Models and roles

| Role | Model | Status |
|---|---|---|
| Standardized patient | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Baseline patient simulator |
| Standardized patient | `Qwen/Qwen3.8-27B-FP8` | Higher-capability comparison |
| Clinician | One model from the primary model cohort | Experimental variable |
| Default clinician | `Qwen/Qwen3.5-122B-A10B-FP8` | Override per run with `--clinician-model` |

The patient simulator remains fixed when clinician models are compared. It is
not counted as a response generator or judge in the primary model cohort.

## Information boundary

| Information | Patient model | Clinician model |
|---|:---:|:---:|
| Private MedSP1000 `sp_actor` packet | Yes | No |
| Private MedSP1000 `examinee` packet | No | Yes |
| Visible conversation so far | Yes | Yes |
| `environment_controller` material | No | No |
| Evaluator, rubric, or checklist material | No | No |
| Other scenarios or reference answers | No | No |

The patient packet is the sole source of patient facts. The clinician receives
only the information that an examinee would have before the encounter, plus
facts the simulated patient subsequently discloses in the visible conversation.

## What the patient model receives

The patient system message contains:

1. The private actor packet for one scenario.
2. Instructions to speak as the patient in ordinary language.
3. Rules against inventing facts or reporting hidden diagnoses, examinations,
   tests, or evaluator content.
4. A requirement to answer concisely and reveal facts incrementally.

After the system message, the patient receives the clean visible conversation
history and the current clinician message. A short grounding reminder is
attached only to the current message: a symptom suggested in a question is not
automatically true, and absent facts must not be filled with plausible details.
Prior copies of this reminder are not retained in history. The patient never
receives the clinician's private initialization.

Conceptually, its input looks like this:

```text
SYSTEM
You are portraying a standardized patient in a realistic text consultation.

PRIVATE PATIENT ACTOR MATERIAL:
<patient_actor_material>
[the selected MedSP1000 actor packet]
</patient_actor_material>

Use only packet facts. Reveal only what is asked or explicitly meant to be
volunteered. Respond naturally in 1–2 short sentences.

USER
<clinician_message>
[the clinician's visible message]
</clinician_message>

[Ground replies in the actor packet; do not adopt suggested details.]
```

## What the clinician model receives

The clinician system message contains:

1. The private examinee, pre-encounter, or triage packet for the same scenario.
2. Instructions that the encounter is text-only.
3. Rules against fabricating examinations, tests, records, or patient facts.
4. The number of available speaking turns and a requirement to follow the
   task and clinical focus in the examinee packet.

After the system message, the clinician receives the clean visible conversation
history and each new patient reply. A current-turn control message tells it how
many speaking turns remain; old copies of that control are not retained. It
never receives the private actor packet.

Conceptually, its input looks like this:

```text
SYSTEM
You are conducting a realistic, text-only clinical consultation.

CLINICIAN-VISIBLE INITIALIZATION:
<clinician_initialization>
[the selected MedSP1000 examinee packet]
</clinician_initialization>

Ask concise, clinically important questions. Do not pretend that an examination
or test occurred. Follow the task in the initialization. You have four speaking
turns.

USER
The patient is ready to speak with you.
```

## Example 1: persistent cough and tobacco use

The private patient input, abridged:

```text
You have had a daily morning cough for roughly two months, sometimes with a
small amount of gray mucus. A prior antibiotic and inhaler did not help. You
smoke one pack per day and have made four previous quit attempts. You doubt
nicotine-replacement medication works. Discuss the detailed smoking history only
as prompted. You are willing to consider another attempt if offered support.
```

The private clinician input, abridged:

```text
You are in a primary-care clinic seeing a 32-year-old with two months of cough.
Two urgent-care visits did not identify another cause, and prior treatment did
not help.
```

Only the clinician knows the pre-encounter framing. Only the patient knows the
detailed smoking habits, earlier quit attempts, concerns, and conditional
responses. In the smoke run, the clinician asked about the cough and smoking;
the patient first disclosed the morning gray mucus and one-pack-per-day use,
then disclosed prior quit attempts when asked.

## Example 2: progressive neck pain

The private patient input, abridged:

```text
You are 65 and have worsening neck pain that is now constant and severe. Your
hands and legs feel weak, you dropped a mug, and you had one episode of urinary
incontinence. You lost about 20 pounds unintentionally. Colon cancer was
resected two years ago, but you did not return for follow-up. Do not give away
all of this information at once.
```

The private clinician input, abridged:

```text
You are a resident on an emergency-department night shift. A 65-year-old patient
presents with neck pain, appears distressed, and has stable supplied vital
signs.
```

The clinician must discover the neurologic and cancer-history red flags through
the conversation. The actor's private packet cannot be used as if it were an
already-visible medical record.

## Example 3: acute productive cough

The private patient input, abridged:

```text
You are 55 and feel acutely ill. Four days ago you had a severe shaking chill;
three days ago a worsening productive cough began. You have right-sided pain
with coughing or a deep breath, exertional shortness of breath, and recent sick
contacts at a shelter. Your opening concern is that you cannot get rid of the
cough. Later in the encounter, you are worried about lung cancer.
```

The private clinician input, abridged:

```text
You are seeing a 55-year-old in primary care for cough. Pre-encounter vital
signs include temperature 101.5 F, respiratory rate 20, and oxygen saturation
95% on room air.
```

The clinician may use the supplied pre-encounter vitals, but must not invent a
lung examination, imaging result, or laboratory result. The patient should not
recite staging directions, props, or hidden teaching instructions.

## Frozen question format

Prepared scenarios are stored as JSON Lines. Each record contains the two
private role inputs and provenance fields:

```json
{
  "schema_version": "2.0",
  "question_id": "mededportal_5102__scenario1",
  "question_type": "multi_turn_standardized_patient",
  "question_text": "private clinician-visible initialization",
  "private_patient_context": "private patient-model context",
  "source_scenario_path": "mededportal_5102/scenario1",
  "question_text_sha256": "...",
  "private_patient_context_sha256": "...",
  "source_revision": "..."
}
```

This follows the repository-wide question-set envelope: `question_text` is what
the model under evaluation initially sees, while task-specific private context
is stored in an explicitly role-restricted field. The full contract is in
`data/question_sets/README.md` and its linked JSON Schema.

The expanded frozen cohort is
`data/question_sets/medsp1000_generation_cases.jsonl`. Its adjacent manifest
records the source revision, selection seed, artifact hash, and exclusion
counts.

## Generation sequence

Each question currently uses four clinician-patient exchanges:

```text
clinician turn 1 -> patient turn 1
clinician turn 2 -> patient turn 2
clinician turn 3 -> patient turn 3
clinician turn 4 -> patient turn 4
```

On every turn, each model receives its own unchanged private system context and
the clean shared visible transcript so far. Internal current-turn controls are
added only for the generation being made and are never saved as dialogue or
carried forward in history. A completed output record is appended only after all
eight visible messages have been generated.

The patient runner uses a 16,384-token model window because the longest actor
packets approach 24,000 characters before conversation history is added. Modal
Qwen clinicians use an 8,192-token window; API clinicians use their provider's
model context capacity. The longest clinician initialization is under 8,000
characters. These are context capacities, not output limits.

## Preparing and running questions

Build the deterministic 200-scenario cohort:

```bash
uv run python scripts/prepare_medsp1000_questions.py
```

Run one infrastructure and realism smoke question:

```bash
uv run modal run src/generation/modal_medsp1000.py --smoke-test
```

The Mistral patient always runs on Modal. The clinician can be a pinned Qwen
model on Modal or an OpenAI, Anthropic/Claude, or Gemini model reached through
the repository's provider-independent inference client. For example:

```bash
uv run modal run src/generation/modal_medsp1000.py \
  --smoke-test \
  --clinician-model claude-sonnet-5
```

Run the expanded cohort after reviewing a small Mistral patient-simulator smoke
set:

```bash
uv run modal run src/generation/modal_medsp1000.py \
  --input-path data/question_sets/medsp1000_generation_cases.jsonl \
  --clinician-model Qwen/Qwen3.5-122B-A10B-FP8 \
  --num-questions 200 \
  --output-path data/outputs/medsp1000/generations.jsonl
```

The runner defaults to the documented Mistral Small 3.1 patient baseline, Qwen
3.5 122B-A10B as the clinician, the complete 200-question set, and
`data/outputs/medsp1000/generations.jsonl`. The requested clinician model and
provider-returned model version are retained in every response.

Conversations are generated in batches of 8 by default. Once a batch is
complete, each conversation row is appended, flushed, and synced individually
before the progress manifest is updated, so a crash cannot cause a
manifest-only success. Restarting the same configuration skips every saved
generation key. Use `--checkpoint-batch-size 1` when minimal rework is more
important than batched throughput; partial conversations are never written.

## Response format

Each JSONL row is one complete conversation attempt. It uses the same
run/generation/attempt/status envelope as the single-turn outputs, with an
ordered `turns` array for the multi-turn response. Each turn records its role,
model, content, exchange and turn indices, finish reason, token usage, and
latency. Row-level totals are also split by clinician and patient.

The exact record contract and a turn example are documented in
`data/outputs/README.md` and
`data/outputs/schemas/medsp1000_multiturn_generation.schema.json`. Every run
also writes a manifest containing the frozen prompt templates and hashes,
question-set hash, model pair, generation parameters, checkpoint progress, and
completion counts. Generation keys include both role-context hashes, the
clinician model, prompt version, exchange count, output caps, sampling
configuration, and seed, so changing any input or setting creates a distinct
resumable result rather than silently reusing an earlier one. The loader
recalculates both role-context hashes before generation.

## Generation-time review

This stage checks realism and data integrity, not clinical-answer quality or
self-preference. Before scaling a configuration, inspect a small smoke set for:

- faithful use of packet facts without unsupported additions;
- incremental rather than wholesale disclosure;
- natural patient language without simulation instructions;
- clinician questions that respond to the conversation rather than a checklist;
- no invented examinations, tests, or records; and
- adherence to the task in the clinician packet, including appropriate handling
  of urgent information when it is relevant to that task.

Evaluator prompts, scoring rubrics, and judge models remain outside this
generation pipeline.
