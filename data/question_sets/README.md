# Question Sets

Question sets are frozen, line-delimited JSON (`.jsonl`) artifacts. Each line is
one independently addressable question or scenario. An adjacent
`.manifest.json` records dataset-level provenance, selection details, the
artifact hash, record count, and a human-readable record schema.

The active artifacts in this directory are the MedSP1000 multi-turn cohort and
the Real-POCQi single-turn set. Historical pilot inputs and intermediate audits
are retained under `data/deprecated/`.

## Common record envelope

New or revised question-set schemas should have:

- `schema_version`: version of that question type's record contract;
- `question_id`: stable and deterministic within the source dataset;
- `question_type`: task shape, such as `single_turn_qa` or
  `multi_turn_standardized_patient`;
- `question_text`: the information presented to the model being evaluated;
- `source_dataset` and `source_revision`: immutable source provenance; and
- a source locator such as `source_file` or `source_scenario_path`.

Task-specific private context must be named explicitly. It must not be folded
into `question_text`, because that would erase the model information boundary.
Some frozen v1 single-turn artifacts predate `question_type`; they remain valid
under the record schema embedded in their own manifests.

## Multi-turn standardized-patient questions

MedSP1000 uses
[`schemas/medsp1000_multi_turn_question.schema.json`](schemas/medsp1000_multi_turn_question.schema.json).

For this task:

- `question_text` is the clinician-visible examinee or triage packet;
- `private_patient_context` is the patient-model-only actor packet;
- `source_scenario_path` retains the stable scenario-level source locator.

The two text fields have separate SHA-256 hashes. Generation configuration,
model identifiers, transcripts, latency, and grading results do not belong in a
question-set record; they belong in generation or evaluation artifacts.

Example:

```json
{
  "schema_version": "2.0",
  "question_id": "mededportal_5102__scenario1",
  "question_type": "multi_turn_standardized_patient",
  "question_text": "You are in primary care seeing a 32-year-old with two months of cough...",
  "private_patient_context": "You have experienced a daily cough for about two months...",
  "source_dataset": "byrLLCC/MedSP1000",
  "source_revision": "55e3e55efd08c73baab912ba0c5b42637114fbc8",
  "source_scenario_path": "mededportal_5102/scenario1",
  "question_text_sha256": "...",
  "private_patient_context_sha256": "...",
  "cohort_index": 1,
  "selection_reason": "strict interactive patient generation cohort",
  "selection_seed": 42,
  "quality_score": 12,
  "official_q100_member": false,
  "history_domains": ["symptom_history", "past_medical_history"]
}
```

Private means role-restricted during generation, not secret on disk. The
question-set artifact contains both texts so the experiment can be reproduced;
the generation runner is responsible for routing each field only to its intended
model. Per-role source file paths are deliberately omitted because the complete
texts and their hashes are already frozen in each row.

## Manifest requirements

Each manifest should contain at least:

- `schema_version`, `artifact`, `artifact_sha256`, and `record_count`;
- `question_type` and `record_schema`;
- immutable source dataset and revision identifiers;
- selection or sampling method and seed, when applicable; and
- any filtering, exclusion, or manual-review notes needed to reproduce and
  interpret the cohort.

JSONL records are written deterministically with one trailing newline. If any
record changes, regenerate the artifact and manifest together so the recorded
hash remains valid.
