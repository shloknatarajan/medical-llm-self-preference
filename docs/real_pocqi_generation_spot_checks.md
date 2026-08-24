# Real-POCQi four-model generation spot checks

QA performed on August 23, 2026 after completing the initial 620-question,
four-model generation run, before the two Qwen models were added. Current
eight-model corpus counts and conclusions are in [`experiment.md`](experiment.md).
This is a research-data spot check, not a substitute for blinded physician
evaluation.

## Corpus integrity

- 2,480 logical generations are present: 620 each for `gpt-5.6-sol`,
  `gpt-5.6-terra`, `claude-opus-5`, and `claude-sonnet-5`.
- All 620 questions and 30 specialties are represented.
- There are no missing keys, unexpected keys, empty successful answers, or
  unresolved failures.
- The append-only file contains 2,484 attempts: 2,480 final successes and four
  superseded Claude Opus truncation attempts.
- All final OpenAI responses report `completed`; all final Anthropic responses
  report `end_turn`.
- No exact duplicate responses or "As an AI" boilerplate were found.

## Response-length profile

| Model | Median output tokens | 5th percentile | 95th percentile | Maximum |
|---|---:|---:|---:|---:|
| `gpt-5.6-sol` | 720 | 284 | 1,753 | 4,519 |
| `gpt-5.6-terra` | 822 | 312 | 1,915 | 3,439 |
| `claude-sonnet-5` | 1,355 | 964 | 1,937 | 5,915 |
| `claude-opus-5` | 2,928 | 1,886 | 4,658 | 8,045 |

Claude Opus is materially more verbose than the other generators. Downstream
judging should account for verbosity/length preference as a possible confound.

Successful attempts used different output caps because the run was safely
resumed after truncation testing: 4,096, 8,192, and (for one Opus answer) 16,384
tokens. Median lengths were similar between the 4,096- and 8,192-cap subsets,
but the differing request parameter should still be recorded as a limitation.

Provider responses recorded the requested model aliases as model versions,
rather than immutable provider snapshot identifiers. The artifact therefore
does not independently prove a pinned backend revision.

## Manual matched-case review

Three questions were selected deterministically from different specialties and
all four answers were reviewed side by side.

### Venous reflux duplex thresholds (Radiology)

Question ID: `68d62cb1-809d-4cfd-8dc8-b30437fa94cb`

- `gpt-5.6-sol` and `gpt-5.6-terra` gave the correct, direct thresholds:
  greater than 500 ms for superficial/perforating veins and greater than 1
  second for common femoral, femoral, and popliteal veins.
- `claude-opus-5` was correct and more technically detailed, including the
  500-ms threshold for tibial and deep femoral veins.
- `claude-sonnet-5` incorrectly grouped superficial veins with femoral and
  popliteal veins at a 1-second threshold. This is a substantive factual error.

Primary check: the 2023 SVS/AVF/AVLS guideline defines reflux as greater than
500 ms in superficial truncal, tibial, deep femoral, and perforating veins, and
greater than 1 second in common femoral, femoral, and popliteal veins:
https://pmc.ncbi.nlm.nih.gov/articles/PMC11523430/

### Genital HSV suppression (Infectious Disease)

Question ID: `23966dfd-2d99-48bb-a270-62213f95434c`

- All four ultimately recommended acyclovir 400 mg orally twice daily for
  suppression and distinguished valacyclovir 500 mg daily as the regimen with
  direct discordant-couple transmission evidence.
- `claude-sonnet-5` initially claimed the pivotal transmission trial used
  acyclovir, then corrected itself in the next section. The final recommendation
  is sound, but the internal contradiction is a meaningful quality defect.
- The other three responses were accurate; Opus was much longer than necessary.

Primary check: CDC genital-herpes guidance:
https://www.cdc.gov/std/treatment-guidelines/herpes.htm

### Selpercatinib plus carboplatin/pemetrexed (Oncology)

Question ID: `cc1e6064-5e27-4b08-a409-ae9600f7c815`

- A randomized phase II trial of carboplatin/pemetrexed with or without
  selpercatinib after progression on prior RET-directed therapy is registered
  as NCT05364645 and is listed by NCI as administratively complete.
- `gpt-5.6-sol`, `gpt-5.6-terra`, and `claude-sonnet-5` correctly said there is
  no established published prospective efficacy/safety dataset, but failed to
  mention the registered phase II trial. Their answers are therefore
  incomplete relative to the exact question.
- `claude-opus-5` also omitted the trial identifier and asserted that small
  safety run-ins/combination experiences exist without naming a verifiable
  source. Treat that claim as unsupported pending source adjudication.

Primary check: NCI trial record for NCT05364645:
https://www.cancer.gov/research/participate/clinical-trials-search/v?id=NCT05364645

## Additional targeted checks

- The shortest Sol response said approximately 30% of OPRA participants had
  clinical N2 rectal cancer, while Terra gave approximately 26% (84/324). This
  numeric disagreement should be source-adjudicated before either answer is
  labeled correct.
- Refusal-like phrase scanning found no true safety refusals. The few matches
  occurred inside otherwise substantive clinical answers (for example,
  explaining that a test cannot establish a diagnosis by itself).

## Assessment

The artifact is structurally complete and suitable for the planned blinded
evaluation. The spot check demonstrates that it should not be treated as a
gold-standard clinical answer set: even strong-looking responses contain
occasional factual errors, internal contradictions, omissions, and unsupported
claims. Those defects are useful signal for the experiment, provided judging is
blinded, source-grounded where feasible, and attentive to verbosity bias.
