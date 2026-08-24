# Documentation

The documentation is organized around a small set of canonical sources:

- [`experiment.md`](experiment.md) is the scientific source of truth: study
  registry, cohorts, methods, completed results, limitations, specialty
  analysis, and artifact map.
- [`medsp1000_generation.md`](medsp1000_generation.md) explains the multi-turn
  information boundary, role prompts, message sequence, and review criteria.
- [`real_pocqi_generation_spot_checks.md`](real_pocqi_generation_spot_checks.md)
  retains detailed manual checks from the initial four-model generation run
  that are too granular for the main experiment report.
- [`latex/med_self_preference/main.tex`](latex/med_self_preference/main.tex) is
  the self-contained LaTeX manuscript source; the compiled PDF is stored at
  `output/pdf/med_self_preference_updated.pdf`.
- [`archive/legacy_manuscript.md`](archive/legacy_manuscript.md) and its
  [original PDF](archive/legacy_manuscript.pdf) preserve the four-model draft.
  Its claims and counts are historical, not the current project conclusions.

Operational documentation stays next to the relevant code or data:

- [`../src/generation/README.md`](../src/generation/README.md): generation and
  judging commands.
- [`../src/inference/README.md`](../src/inference/README.md): unified inference
  API.
- [`../data/question_sets/README.md`](../data/question_sets/README.md): frozen
  question-set contracts.
- [`../data/outputs/README.md`](../data/outputs/README.md): generation record
  format.
- [`../data/analysis/self_preference/README.md`](../data/analysis/self_preference/README.md):
  derived score definitions and reproduction commands.

Intermediate result snapshots and summary notes were removed after their useful
content was reconciled into `experiment.md`. This avoids conflicting counts as
append-only runs progress.
