# Deprecated data artifacts

These frozen artifacts are retained for provenance and historical reference,
but they are not active experiment question sets:

- `medsp1000_generation_pilot_cases.jsonl`: the earlier five-case curated
  MedSP1000 pilot input;
- `medsp1000_scenario_audit.jsonl`: the intermediate MedSP1000 screening audit
  used to construct the active cohort; and
- `chatdoctor_healthcaremagic_questions.jsonl`: the earlier ChatDoctor patient
  simulator pilot input;
- `medsp1000_v1_smoke/`: the obsolete one-case MedSP1000 generation made with
  the v1 prompts. It is retained for provenance only because its patient reply
  contained unsupported details and its clinician departed from the assigned
  task.

Each JSONL artifact remains paired with its original manifest. New experiment
runs should use `data/question_sets/medsp1000_generation_cases.jsonl` or another
explicitly active question set instead.
