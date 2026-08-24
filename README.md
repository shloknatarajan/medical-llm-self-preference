# Medical LLM self-preference

This repository contains the experiment code, frozen inputs, generated model
outputs, judgments, and analysis artifacts for studying whether medical large
language models favor outputs from their own model family.

The experiments cover two settings:

- **Real-POCQi:** single-turn specialist question answering.
- **MedSP1000:** multi-turn standardized-patient conversations evaluated at
  multiple conversation lengths.

The repository intentionally retains append-only generation and judgment
records, including failed attempts and run manifests, so the reported analyses
can be audited from the saved model outputs rather than only from aggregate
tables.

## Repository layout

- `src/generation/`: Real-POCQi and MedSP1000 generation pipelines.
- `src/inference/`: a common OpenAI, Anthropic, Gemini, and Modal inference
  interface.
- `src/judging/`: blinded and identity-revealed judging pipelines.
- `scripts/`: question preparation and analysis commands.
- `data/question_sets/`: frozen experiment inputs and provenance manifests.
- `data/outputs/` and `data/real_pcoqi/`: generated responses, judgments, and
  run manifests.
- `data/analysis/`: derived, reproducible analysis tables.
- `data/deprecated/`: historical inputs and pilot outputs retained only for
  provenance.
- `docs/`: experiment design, quality assurance, and result writeups.
- `leaderboard/`: static results explorer.

See [`src/generation/README.md`](src/generation/README.md) for generation and
judging commands, [`data/outputs/README.md`](data/outputs/README.md) for the
saved-output format, and
[`data/analysis/self_preference/README.md`](data/analysis/self_preference/README.md)
for analysis definitions.

## Setup

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync --dev
uv run pytest -q
```

Provider credentials are read from environment variables or an ignored `.env`
file. Depending on the selected cohort, the pipelines use `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, and Modal's standard authentication.
Never commit credential files.

## Reproduce the analyses

The committed raw outputs are sufficient to regenerate the derived
self-preference tables without making provider API calls:

```bash
uv run python scripts/create_self_preference_scores.py
uv run python scripts/analyze_new_self_preference.py
uv run python scripts/analyze_token_length_confounding.py
```

The scripts record source and output hashes in their manifests. Large JSONL
artifacts are stored with Git LFS; clone with Git LFS enabled to obtain their
contents.

## Data provenance

Active question sets are frozen under `data/question_sets/` with source
revisions, selection details, schemas, and content hashes. Downloaded upstream
source snapshots under `data/source/` are deliberately ignored because they
are re-downloadable and may contain nested repositories. Historical experiment
artifacts are not deleted; they are labeled and retained under
`data/deprecated/`.

Model outputs are research artifacts and may contain inaccurate or unsafe
medical content. They are not medical advice and should not be used for
clinical decision-making.
