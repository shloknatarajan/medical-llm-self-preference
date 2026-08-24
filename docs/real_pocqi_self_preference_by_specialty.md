# Real-POCQi self-preference by clinical specialty

Analysis performed August 24, 2026 on the completed single-turn Real-POCQi
combined judging condition. The main result is a strong, model-dependent
self-preference signal that is broadly stable across clinical specialties. The
data do **not** provide convincing evidence that self-preference is concentrated
in particular specialties.

## Scope and estimand

This analysis uses the latest successful record for each logical
judge-question key from
`data/real_pcoqi/judgements/rubric_and_model_ranking.jsonl`, restricted to
experiment `real_pocqi_combined_all_judges_v1`. The resulting dataset contains
4,960 judgments: 620 questions, eight judges, eight candidate generators, and
30 values of the source dataset's `specialty` field.

The primary outcome is a matched rank difference. For judge \(j\) and question
\(q\), it is

```text
rank assigned by j to j's own response
  - mean rank assigned by the other seven judges to that same response.
```

Lower ranks are better, so a negative difference indicates self-preference.
This comparison holds the question and candidate answer fixed. It therefore
does not mistake a generally strong generator for self-preference: a judge gets
credit only for ranking its own answer more favorably than the other judges
rank that exact answer.

Candidate order varied across judge-question records and had a measurable
effect. Ranks were therefore adjusted using a pooled model with
judge-by-generator fixed effects and presentation-position effects. The
position-adjusted and unadjusted self-preference estimates were nearly
identical, indicating that random order imbalance does not explain the main
finding. Confidence intervals below are descriptive normal-approximation 95%
intervals over questions; pooled intervals first average the eight correlated
judge observations within each question.

## Main findings

Across all judges and questions, a judge placed its own answer **1.219 rank
positions higher** than the other judges did for the same answer (adjusted
difference −1.219, 95% CI −1.261 to −1.177). The corresponding same-family
preference was −1.269 positions (95% CI −1.311 to −1.227).

The effect differs far more by judge than by specialty. The two GPT judges have
the largest matched effects. Gemini Flash also has a substantial effect that
mostly changes middle ranks rather than first-place selection. Opus almost
always ranks its answer first, but its matched effect is smaller than the GPT
effects because other judges already rank the Opus answer highly.

| Judge | Raw mean rank of own answer | Adjusted self-preference | 95% CI | First-place uplift |
|---|---:|---:|---:|---:|
| `gpt-5.6-sol` | 1.735 | −2.939 | [−3.041, −2.836] | +33.6 pp |
| `gpt-5.6-terra` | 1.442 | −2.224 | [−2.325, −2.123] | +56.5 pp |
| `gemini-3.7-flash` | 2.844 | −1.236 | [−1.317, −1.155] | +0.9 pp |
| `claude-opus-5` | 1.040 | −0.969 | [−1.019, −0.919] | +32.3 pp |
| `Qwen/Qwen3.5-122B-A10B-FP8` | 5.884 | −0.730 | [−0.852, −0.608] | +2.5 pp |
| `gemini-3.1-pro-preview` | 4.023 | −0.683 | [−0.790, −0.575] | +1.3 pp |
| `claude-sonnet-5` | 3.816 | −0.647 | [−0.763, −0.530] | −0.3 pp |
| `Qwen/Qwen3.8-27B-FP8` | 6.682 | −0.324 | [−0.419, −0.228] | −0.4 pp |

First-place uplift is the judge's own first-place rate minus the mean rate at
which the other seven judges put that answer first. It is unadjusted for
position and is secondary to the full-rank outcome.

The Qwen results illustrate why raw own-answer rank or first-place rate alone
is insufficient. Both Qwen answers remain weak in absolute terms, but each
Qwen judge ranks its own answer more favorably than the other judges do. This
is a relative self-preference signal even when it rarely produces a win.

## Specialty-level results

Among specialties with at least 10 questions, adjusted pooled estimates range
from −1.543 positions in Family Medicine to −0.908 in Radiology. All are
negative, but their ordering should not be interpreted as a set of confirmed
specialty differences.

| Specialty | Questions | Explicit-rank self-preference | 95% CI | Rubric-sum-rank sensitivity | Same-family preference |
|---|---:|---:|---:|---:|---:|
| Endocrinology | 55 | −1.359 | [−1.507, −1.211] | −1.320 | −1.399 |
| Cardiology | 54 | −1.189 | [−1.329, −1.049] | −1.064 | −1.297 |
| Oncology / Hematology | 49 | −1.052 | [−1.176, −0.927] | −1.029 | −1.116 |
| OB / GYN | 44 | −1.283 | [−1.441, −1.126] | −1.216 | −1.344 |
| Dermatology | 40 | −1.097 | [−1.268, −0.927] | −1.113 | −1.185 |
| Rheumatology | 39 | −1.234 | [−1.400, −1.069] | −1.166 | −1.287 |
| Neurology | 32 | −1.231 | [−1.415, −1.047] | −1.158 | −1.209 |
| Infectious Disease | 30 | −1.322 | [−1.505, −1.138] | −1.348 | −1.339 |
| Psychiatry | 25 | −1.359 | [−1.534, −1.184] | −1.244 | −1.355 |
| Surgery | 25 | −1.371 | [−1.635, −1.106] | −1.225 | −1.446 |
| Nephrology | 20 | −1.172 | [−1.474, −0.869] | −1.048 | −1.260 |
| Allergy & Immunology | 19 | −1.249 | [−1.527, −0.970] | −1.128 | −1.290 |
| Pediatrics | 19 | −1.188 | [−1.395, −0.980] | −1.193 | −1.208 |
| Internal Medicine | 17 | −1.256 | [−1.536, −0.977] | −1.235 | −1.286 |
| Critical Care | 14 | −1.221 | [−1.463, −0.980] | −1.197 | −1.247 |
| Gastroenterology | 14 | −1.141 | [−1.404, −0.878] | −1.101 | −1.220 |
| Pulmonology | 14 | −1.197 | [−1.398, −0.997] | −1.065 | −1.148 |
| Transplantation | 14 | −1.140 | [−1.366, −0.913] | −1.044 | −1.287 |
| Anesthesiology | 11 | −1.297 | [−1.606, −0.989] | −1.091 | −1.241 |
| Family Medicine | 11 | −1.543 | [−1.902, −1.184] | −1.308 | −1.542 |
| Orthopedics | 11 | −1.215 | [−1.502, −0.928] | −1.265 | −1.332 |
| Primary Care | 11 | −1.397 | [−1.821, −0.974] | −1.258 | −1.334 |
| Emergency Medicine | 10 | −1.243 | [−1.560, −0.925] | −1.232 | −1.288 |
| Radiology | 10 | −0.908 | [−1.186, −0.630] | −1.023 | −1.034 |

Six specialties have fewer than 10 questions and are too sparse for useful
comparisons. Their descriptive explicit-rank estimates are Urology (8,
−1.119), Hospital-Based Medicine (7, −0.862), Surgical Oncology (6, −1.063),
Genetics (4, −0.684), Otolaryngology (4, −0.933), and Ophthalmology (3,
−1.056).

## Is there evidence of specialty heterogeneity?

A label-permutation test was run on the question-level pooled adjusted effect,
preserving the 30 observed specialty sizes. Specialty accounted for 5.7% of
the question-level variation, but the global test was not significant
(`p=0.190`, 10,000 permutations). In other words, the observed spread across
specialties is compatible with sampling variation across many unequally sized
groups.

Separate permutation tests also found no significant specialty heterogeneity
for any judge:

| Judge | Variance explained by specialty | Permutation p-value |
|---|---:|---:|
| `claude-sonnet-5` | 6.2% | 0.103 |
| `Qwen/Qwen3.8-27B-FP8` | 5.3% | 0.281 |
| `gemini-3.7-flash` | 4.9% | 0.406 |
| `gpt-5.6-terra` | 4.8% | 0.439 |
| `gemini-3.1-pro-preview` | 4.7% | 0.460 |
| `gpt-5.6-sol` | 4.6% | 0.491 |
| `Qwen/Qwen3.5-122B-A10B-FP8` | 3.8% | 0.773 |
| `claude-opus-5` | 3.1% | 0.918 |

These are diagnostic interaction tests, not eight independent confirmatory
hypotheses. None crosses a conventional 0.05 threshold even before correction
for multiple testing.

## Robustness checks

- **Presentation order:** adjusting for response position changes the overall
  estimate from −1.219 to −1.219 after rounding. Judge-specific changes are
  also small; for example, the Sol estimate changes from −2.932 to −2.939.
- **Rubric-derived ranking:** ranking candidates by their five-axis score sums
  yields the same direction for every judge. Across the 30 specialty estimates,
  explicit-rank and rubric-sum-rank effects correlate at `r=0.889`.
- **Family preference:** each judge also ranks the two responses from its model
  family more favorably than judges outside that family do. The pooled
  same-family estimate (−1.269) is close to the exact-model estimate (−1.219),
  suggesting that shared model-family style or evaluation criteria contribute
  substantially to the measured effect.

## Interpretation and limitations

The best-supported conclusion is that self-preference is a general property of
these judges, with large differences in magnitude by model but no reliable
subspecialty interaction in this sample. The specialty labels are broad and
highly imbalanced (3 to 55 questions), so absence of detected heterogeneity is
not proof that clinically narrower domains have identical effects.

Generator identities were hidden from the judges. Accordingly, “self-
preference” here is an operational behavioral result, not evidence that a model
recognized its own output. It may reflect alignment with its own prose style,
reasoning conventions, favored evidence, or family-specific quality criteria.
The same-family result makes that explanation especially plausible.

The analysis is descriptive and does not adjust the specialty-specific
confidence intervals for 30 comparisons. A confirmatory follow-up should
pre-specify a smaller set of adequately powered specialties, counterbalance
candidate order, and use a hierarchical rank model with question, judge,
generator, family, and specialty interactions. Repeated generations would also
be needed to separate stable model-family affinity from idiosyncrasies of one
answer per model-question pair.

## Bottom line

There is strong matched self-preference in the single-turn QA results, including
for models whose answers rarely rank first. The magnitude is driven primarily
by which model is judging—especially the two GPT judges—not by the clinical
specialty. Current specialty differences are exploratory and should not be
reported as established subspecialty effects.
