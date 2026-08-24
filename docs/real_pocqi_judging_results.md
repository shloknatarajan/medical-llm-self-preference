# Real-POCQi judging run and result QA

QA performed on August 24, 2026 after completing the combined rubric-scoring
and model-ranking condition. This report describes execution integrity and
initial descriptive results; it is not a substitute for the planned
statistical analysis.

## Run design

- The corpus contains 620 questions and eight candidate responses per question.
- Six non-Modal models judged every question: `gpt-5.6-sol`,
  `gpt-5.6-terra`, `claude-opus-5`, `claude-sonnet-5`,
  `gemini-3.1-pro-preview`, and `gemini-3.7-flash`.
- Candidates were presented under blinded identifiers (`response-1` through
  `response-8`). Generator names were retained only in the saved provenance
  mapping used to resolve rankings after inference.
- Each judge assigned an absolute 0--5 score on the five original Real-POCQi
  axes: accuracy, clinical utility, source quality, verifiability, and
  completeness. The judge then produced an explicit best-to-worst ranking.
- The production experiment ID is `real_pocqi_combined_all_judges_v1`.
- Results are stored in
  `data/real_pcoqi/judgements/rubric_and_model_ranking.jsonl`. The file is
  append-only and also contains failed-attempt audit history and earlier smoke
  tests, so production analyses must filter on the experiment ID and successful
  status rather than treating every physical line as an observation.

## Execution integrity

The completed artifact passed the following checks:

- 3,720 successful logical judgments are present: 620 questions times six
  judges.
- All 3,720 judgment keys are unique, with exactly 620 successes from each
  judge and exactly six judges for each question.
- All 29,760 candidate records contain eight distinct response IDs, generation
  IDs, and generator models.
- All 148,800 axis scores parse successfully and fall within the required 0--5
  interval.
- Every explicit model ranking and deterministic score-sum ranking is a valid
  permutation of the eight candidates and resolves back to the saved generator
  provenance.
- No explicit generator-model name occurs in any saved system or user prompt.
- Every final successful request has a normal provider finish reason:
  `completed` for OpenAI, `end_turn` for Anthropic, and `STOP` for Gemini.
- A resumability dry run reports 0 pending and 3,720 skipped judgments.
- The automated suite passes with 67 tests.

The initial 4,096-token smoke configuration exposed truncated structured
outputs. The production sweep used an 8,192-token cap. Five Sonnet judgments
and one Gemini Pro judgment still required a targeted 16,384-token cleanup
pass; all six succeeded on that pass. These were reasoning/output-budget
failures, not missing inputs or rate-limit failures.

## Aggregate response performance

Lower mean rank is better. Percent ranked first is computed over all 3,720
judge-question decisions for each generator.

| Generator | Mean rank | Ranked first | Mean rubric sum (of 25) |
|---|---:|---:|---:|
| `claude-opus-5` | 2.073 | 65.8% | 22.525 |
| `gpt-5.6-terra` | 3.180 | 21.3% | 21.165 |
| `gemini-3.7-flash` | 3.905 | 0.9% | 20.549 |
| `gpt-5.6-sol` | 4.058 | 10.5% | 20.440 |
| `claude-sonnet-5` | 4.433 | 0.5% | 20.050 |
| `gemini-3.1-pro-preview` | 4.591 | 0.8% | 19.935 |
| `Qwen/Qwen3.5-122B-A10B-FP8` | 6.654 | 0.1% | 17.038 |
| `Qwen/Qwen3.8-27B-FP8` | 7.107 | 0.1% | 16.012 |

The explicit model ranking and deterministic score-sum ranking have 94.0%
pairwise ordering agreement. Their complete eight-model order is identical in
27.6% of judgments. This is not a schema inconsistency: the combined prompt
asks the judge to complete the rubric and then rank responses directly, without
requiring the explicit ranking to equal the score sum.

## Judge calibration

Absolute-score calibration differs substantially across judges.

| Judge | Mean axis score | Median | Scores exactly 5 |
|---|---:|---:|---:|
| `claude-opus-5` | 3.786 | 4.0 | 7.2% |
| `claude-sonnet-5` | 3.891 | 4.0 | 10.0% |
| `gemini-3.1-pro-preview` | 4.345 | 5.0 | 54.9% |
| `gemini-3.7-flash` | 4.491 | 4.7 | 21.0% |
| `gpt-5.6-sol` | 3.981 | 4.2 | 2.9% |
| `gpt-5.6-terra` | 3.163 | 3.6 | 3.6% |

Consequently, raw absolute scores should not be pooled across judges as though
their scales were interchangeable. Use within-judge normalization, judge fixed
effects, or an appropriate hierarchical model for cross-judge comparisons.

## Self-preference and judge-family effects

| Judge | Mean rank of its own response | Own response ranked first |
|---|---:|---:|
| `claude-opus-5` | 1.040 | 97.6% |
| `claude-sonnet-5` | 3.816 | 1.3% |
| `gemini-3.1-pro-preview` | 4.023 | 2.4% |
| `gemini-3.7-flash` | 2.844 | 2.1% |
| `gpt-5.6-sol` | 1.735 | 37.7% |
| `gpt-5.6-terra` | 1.442 | 66.9% |

The pattern is not uniform self-preference. Both GPT judges strongly favor GPT
responses, while Claude and Gemini judges generally rank the Opus response
first. Opus itself ranks its own response first in 97.6% of cases. These are
substantive judge/family effects rather than evidence of a pipeline failure,
because generator identities were not present in the prompts and all ranking
permutations validate.

## Presentation-order effect

Candidate ordering was randomized by question and reasonably distributed over
the eight generator models. Nevertheless, an adjusted descriptive check found
a residual preference for earlier positions after controlling for the
judge-generator combination:

| Response position | Adjusted rank residual |
|---:|---:|
| 1 | -0.313 |
| 2 | -0.157 |
| 3 | -0.070 |
| 4 | -0.001 |
| 5 | +0.097 |
| 6 | +0.159 |
| 7 | +0.184 |
| 8 | +0.100 |

Negative values indicate a better rank than the judge-generator baseline.
Randomization limits systematic confounding of a particular generator, but the
final analysis should include response position as a covariate. A robustness
run with counterbalanced or repeated orderings would provide a stronger test of
position sensitivity.

## Assessment

The judging artifact is structurally complete, correctly blinded, resumable,
and suitable for analysis. The aggregate results are coherent: strong models
rank ahead of the two Qwen baselines, explicit rankings closely track rubric
sums, and all providers produce valid structured outputs.

The main scientific caveats are judge-specific score calibration, strong
judge-family/self-preference effects, and measurable presentation-order bias.
Recommended primary analyses should therefore:

1. analyze explicit rankings separately from deterministic rubric-sum ranks;
2. normalize rubric scores within judge or model judge effects explicitly;
3. report results by judge as well as pooled across judges;
4. control for candidate presentation position; and
5. treat self-preference as a measured outcome, not merely a nuisance variable.
