# Med-Self-Preference: Updated Draft and Complete Experiment Compendium

Zara Ansari<sup>1,2</sup>, Shlok Natarajan<sup>1</sup>, Author Two<sup>1</sup>, Author Three<sup>1</sup>, Aaron Fanous<sup>1</sup>, Roxana Daneshjou<sup>1,3</sup>

<sup>1</sup> Department of Biomedical Data Science, <sup>3</sup> Department of Dermatology, Stanford University, 450 Jane Stanford Way, Stanford, CA 94305, USA.<br>
<sup>2</sup> Department of Computer Science, Harvey Mudd College, 301 Platt Boulevard, Claremont, CA 91711, USA.

Emails: zansari6@stanford.edu, insert email, author2@stanford.edu, author3@stanford.edu, author4@stanford.edu, roxanad@stanford.edu

Updated August 24, 2026. This document was derived from
`Med_Self_Preference_First_Draft.md` and reconciled against the result artifacts
and QA reports in this repository. Final production-result counts are current
through 12:25 PDT on August 24, 2026.

## Abstract

Large language models (LLMs) are increasingly used to judge medical-model
outputs, but they may favor responses produced by themselves or by related
models. We report two phases of medical LLM self-preference experiments. The
legacy phase used four models in 125-question Real-POCQi single-turn and
125-scenario ChatDoctor multi-turn pairwise evaluations; its findings are
retained from the first draft, although its raw judgments are not present in
this repository. The expanded phase uses eight generators and judges on all 620
Real-POCQi questions and six clinician generators and judges on 200 MedSP1000
standardized-patient scenarios. In the completed blinded Real-POCQi condition,
eight judges produced 4,960 rubric-plus-ranking judgments over eight candidate
answers. A matched analysis found that judges placed their own answers 1.219
rank positions higher than other judges placed the same answers. The effect was
largest for `gpt-5.6-sol` (−2.939 positions) and `gpt-5.6-terra` (−2.224), but
was negative for all eight judges. A draft-compatible raw-score analysis found
SP-Bias values from −0.749 to +0.668 points on the 0–5 scale, while listwise
own-pick rates ranged from 18.8% to 99.4%, showing that the measures are not
interchangeable. A no-rubric ranking condition on 100 questions
closely reproduced the aggregate order. Specialty explained only 5.7% of
question-level self-preference variation and did not show significant global
heterogeneity (`p=0.190`). In MedSP1000, the complete conditions at all four
transcript lengths show matched self-preference, but the effect is not monotonic
with conversation length. Longer answers score better conditionally, although
length moderation reverses direction across the two tasks and does not explain
away the matched effect. The completed
identity-revealed Real-POCQi follow-up
finds slightly weaker matched self-preference with generator names visible
(-1.391 versus -1.514 positions when blinded; paired change +0.123, 95% CI
+0.059 to +0.188). Overall, self-preference
is robust but highly judge-dependent, and raw win rates alone can obscure it in
weaker generators.

## 1. Motivation

Human review of thousands of medical-model outputs is expensive and slow, so
LLMs are now frequently used as scalable evaluators [Zheng et al., 2023; Li et
al., 2026]. This creates a circularity risk when a model judges outputs from
itself or a related family. Prior work has identified positional, verbosity,
style, and self-recognition effects in general-purpose and medical LLM judging
[Panickssery et al., 2024; Wataoka et al., 2025; Alvarez-Arenas et al., 2026;
Pombal et al., 2026]. In medicine, a biased evaluator can distort conclusions
about clinical accuracy, utility, and safety.

The experiments in this project ask four related questions:

1. Do medical LLM judges rank or score their own outputs more favorably than
   other judges rank or score the same outputs?
2. Does this behavior depend on the judge, model family, rubric, or whether
   generator identities are visible?
3. Does it vary across clinical specialties?
4. Does it change as a standardized-patient conversation becomes longer?

## 2. Experiment registry

The project contains a legacy four-model phase and a newer expanded-cohort
phase. “Complete” means the intended logical matrix for that condition is
present after deduplicating retries; it does not mean that every planned future
extension has been run.

| Phase | Experiment | Intended scale | Status at cutoff | Primary finding |
|---|---|---:|---|---|
| Legacy | Four-model Real-POCQi pairwise QA | 1,500 judgments | Complete as reported; raw artifact unavailable here | GPT had very high own-pick rates; score-level effects differed by model |
| Legacy | Four-model ChatDoctor multi-turn judging | 6,000 judgments | Complete as reported; raw artifact unavailable here | Reported self-preference often strengthened at longer lengths, but patterns were model- and metric-specific |
| Development | Ungrounded patient-simulator comparison | 50 paired cases, 300 patient turns | Generation complete; blinded human review not completed | Both simulators completed all turns; Mistral was faster |
| Development | Reference-grounded patient-simulator comparison | 50 paired cases, 300 patient turns | Generation complete; blinded human review not completed | Both simulators completed all turns; Mistral was faster but more verbose |
| Development | MedSP1000 one-case generation pilots | 1 case per patient simulator | Complete | Confirmed role separation and end-to-end generation; not an efficacy comparison |
| Expanded | Eight-model Real-POCQi generation | 620 × 8 = 4,960 answers | Complete | All cells present; Opus was much more verbose than the other generators |
| Expanded | Blinded Real-POCQi rubric plus ranking | 620 × 8 = 4,960 judgments | Complete | All judges showed matched self-preference; Opus ranked best overall |
| Expanded | Qwen-judge extension to the combined condition | 620 × 2 = 1,240 judgments | Complete and incorporated above | Qwen judges preserved the aggregate model order; matched analysis still detects relative self-preference |
| Expanded | Blinded no-rubric direct ranking | 100 × 8 = 800 judgments | Complete | Closely agrees with combined-condition rankings |
| Expanded | Real-POCQi specialty analysis | 30 specialties | Complete derived analysis | No significant specialty heterogeneity |
| Expanded | Identity-revealed Real-POCQi follow-up | 200 × 6 = 1,200 judgments | Complete | Generator names modestly reduced rather than increased matched self-preference: revealed-minus-blinded +0.123 positions [95% CI +0.059, +0.188] |
| Expanded | Six-model MedSP1000 generation | 200 × 6 = 1,200 conversations | Complete | Full six-model matrix available; planned Qwen clinician generation has not been run |
| Expanded | MedSP1000 combined judging, 2 turns | 200 × 6 = 1,200 judgments | Complete | Pooled matched self-preference −0.518 positions |
| Expanded | MedSP1000 combined judging, 4 turns | 200 × 6 = 1,200 judgments | Complete | Pooled matched self-preference −0.457 positions |
| Expanded | MedSP1000 combined judging, 6 turns | 200 × 6 = 1,200 judgments | Complete | Pooled matched self-preference −0.502 positions |
| Expanded | MedSP1000 combined judging, 8 turns | 200 × 6 = 1,200 judgments | Complete; early same-condition batches are incorporated | Pooled matched self-preference −0.613 positions |

Infrastructure smoke runs, retries, cleanup passes, and the deprecated
MedSP1000 v1 smoke are documented in the artifacts but are not treated as
separate scientific experiments. The v1 smoke was retired because its patient
invented unsupported details and its clinician departed from the assigned task.

Excluding standalone pilot conditions and derived subanalyses, the expanded phase has 11,760
completed production judgments: 4,960 combined Real-POCQi judgments, 800 direct
Real-POCQi judgments, 1,200 identity-revealed Real-POCQi judgments, and 4,800
MedSP1000 judgments across four lengths.

## 3. Data and model cohorts

### 3.1 Legacy four-model cohort

The first draft reports experiments with GPT-5.5, Claude Sonnet 5, Gemini 3.1
Flash-Lite, and Qwen 3.6 35B. Each model served as generator and judge. The
single-turn phase sampled 125 Real-POCQi questions. The multi-turn phase sampled
125 ChatDoctor-HealthCareMagic scenarios and evaluated 2-, 4-, 6-, and 8-turn
truncations. The draft reports 7,500 identity-blind, order-randomized pairwise
judgments in total.

The legacy sample size was based on a two-sided 0.05 significance level and 90%
power. The first draft reports that approximately 119 paired observations were
needed for a score-level effect of `d=0.30`, and approximately 114 observations
were needed to detect a 65% own-pick rate against a 50% null. The selected 125
scenarios allowed limited attrition.

The raw legacy generations and judgments are not in the present repository;
searching the data directory finds those model identifiers only in the first
draft. Consequently, the legacy tables below are a faithful carry-forward of
reported results, not a fresh reanalysis. They should not be pooled with the
expanded experiments.

### 3.2 Expanded Real-POCQi cohort

The frozen Real-POCQi artifact contains all 620 questions from source revision
`9002e1ddff506d354f1b7becc1213b96299d07f6`, covering 30 specialty labels. The
expanded cohort has two models from each of four families:

| Family | Higher-capability tier | Workhorse tier |
|---|---|---|
| OpenAI | `gpt-5.6-sol` | `gpt-5.6-terra` |
| Anthropic | `claude-opus-5` | `claude-sonnet-5` |
| Google | `gemini-3.1-pro-preview` | `gemini-3.7-flash` |
| Qwen | `Qwen/Qwen3.5-122B-A10B-FP8` | `Qwen/Qwen3.8-27B-FP8` |

All eight models generated responses and judged the complete blinded condition.
The completed identity-revealed follow-up uses the six API judges and retains
all eight generators.

### 3.3 Expanded MedSP1000 cohort

The active MedSP1000 artifact contains 200 deterministic scenarios selected
from source revision `55e3e55efd08c73baab912ba0c5b42637114fbc8` with seed
42. Each scenario has separate clinician-visible initialization and private
standardized-patient actor material. Evaluator and environment-controller
materials are excluded.

The patient simulator is fixed to
`mistralai/Mistral-Small-3.1-24B-Instruct-2503`. Six API models—Sol, Terra,
Opus, Sonnet, Gemini Pro, and Gemini Flash—have complete clinician generations
and are the six candidate generators and judges in the current multi-turn
results. Although the broader cohort document proposes two Qwen clinicians and
judges, full MedSP1000 Qwen generations and judgments have not yet been run.

## 4. Evaluation and analysis

### 4.1 Legacy pairwise estimands

The legacy experiments used five 0–5 dimensions: faithfulness, completeness,
safety, clarity, and conciseness. Score-level SP-Bias held one output fixed and
subtracted an outside judge's overall score from the own-family judge's score.
Decision-level self-preference was the proportion of non-tied pairwise choices
in which the judge selected its own model. The draft used paired one-sample
t-tests for score differences and two-sided binomial tests against 50% for
pairwise choices.

### 4.2 Expanded listwise estimand

The expanded combined prompt asks every judge to score every candidate on
accuracy, clinical utility, source quality, verifiability, and completeness,
then independently rank the full candidate list. The primary self-preference
measure in this update is matched exact-model rank difference:

```text
rank assigned by judge j to j's answer
  - mean rank assigned by the other judges to the same answer and question.
```

Negative values indicate preference for the judge's own response. Unlike a raw
own-answer rank, this estimand controls for the answer's general quality by
holding the answer and question fixed. Candidate presentation position is
adjusted using a pooled additive model with judge-by-generator fixed effects.
Uncertainty intervals are descriptive 95% normal-approximation intervals over
questions. The expanded analyses remain exploratory rather than preregistered
confirmatory tests.

### 4.3 Draft-compatible score-level and decision-level measures

To connect the expanded experiments directly to the first draft, we also
compute its two original measures from the complete listwise judgments. Let
`s_kqj` be judge `k`'s mean five-axis score for model `j`'s answer to question
`q`. The generalized score-level measure is:

```text
SP-Bias_jq = s_jqj - mean(s_kqj for every outside judge k).
```

Positive values indicate own-judge score inflation and negative values indicate
self-criticism. Directional model-vs-model results use one outside judge at a
time and exactly reproduce the first draft's estimand. Means, paired t
intervals, and paired one-sample t-tests are calculated across questions.

The strict full-list rankings contain no ties. We convert them into all
head-to-head decisions involving the judge's own answer. The model-level
own-pick rate is the fraction of competitors placed below the own answer,
aggregated across questions. Pairwise rows use the first draft's two-sided
binomial test against 50%; pooled model-level binomial results are descriptive
because comparisons against different competitors within a question are
correlated. These raw-score and own-pick measures are reported as
draft-compatible secondary analyses. They do not replace the position-adjusted
matched-rank primary analysis, and raw SP-Bias should be interpreted cautiously
because judges use the absolute rubric scale differently.

## 5. Legacy four-model experiments

### 5.1 Legacy Real-POCQi single-turn results

The first draft reports that Gemini and Qwen showed the clearest score-level
generosity toward their own answers. GPT showed comparatively little numeric
inflation but chose its own answer in 96–99% of its pairwise judgments. Claude
was frequently self-critical relative to Gemini and Qwen.

| Model answer | Own judge | Other judge | Reported SP-Bias | 95% CI | p |
|---|---|---|---:|---|---:|
| Claude Sonnet 5 | Claude | GPT-5.5 | +0.17 | [+0.10, +0.23] | <0.001 |
| Claude Sonnet 5 | Claude | Gemini | −0.49 | [−0.57, −0.40] | <0.001 |
| Claude Sonnet 5 | Claude | Qwen | −0.12 | [−0.23, −0.02] | 0.022 |
| GPT-5.5 | GPT | Claude | +0.23 | [+0.18, +0.28] | <0.001 |
| GPT-5.5 | GPT | Gemini | −0.02 | [−0.08, +0.04] | 0.555 |
| GPT-5.5 | GPT | Qwen | +0.07 | [−0.01, +0.14] | 0.084 |
| Gemini 3.1 Flash-Lite | Gemini | Claude | +0.29 | [+0.22, +0.37] | <0.001 |
| Gemini 3.1 Flash-Lite | Gemini | GPT | +0.59 | [+0.53, +0.65] | <0.001 |
| Gemini 3.1 Flash-Lite | Gemini | Qwen | +0.12 | [+0.06, +0.18] | <0.001 |
| Qwen 3.6 35B | Qwen | Claude | +0.22 | [+0.11, +0.32] | <0.001 |
| Qwen 3.6 35B | Qwen | GPT | +0.63 | [+0.54, +0.72] | <0.001 |
| Qwen 3.6 35B | Qwen | Gemini | −0.14 | [−0.23, −0.04] | 0.004 |

| Judge | Opponent | Reported own-pick rate | Other | Tie |
|---|---|---:|---:|---:|
| Claude | GPT | 31% | 67% | 2% |
| Claude | Gemini | 30% | 66% | 3% |
| Claude | Qwen | 46% | 48% | 6% |
| GPT | Claude | 97% | 2% | 1% |
| GPT | Gemini | 96% | 3% | 1% |
| GPT | Qwen | 99% | 1% | 0% |
| Gemini | Claude | 57% | 40% | 3% |
| Gemini | GPT | 36% | 58% | 6% |
| Gemini | Qwen | 58% | 39% | 2% |
| Qwen | Claude | 49% | 49% | 2% |
| Qwen | GPT | 26% | 71% | 2% |
| Qwen | Gemini | 42% | 54% | 4% |

These two measures answer different questions. The score-level estimand holds
the answer fixed and is better isolated from output quality. The pairwise own-
pick rate can reflect both self-preference and a genuinely stronger answer.

### 5.2 Legacy ChatDoctor multi-turn results

The reported score-level results were not uniformly monotonic, but many effects
became larger at longer transcript lengths. Gemini scored its own conversations
higher than all three outside judges at every length. Qwen generally did so
against Claude and GPT. Claude was self-critical against Gemini and Qwen, while
its comparison with GPT changed sign.

| Model answer; outside judge | 2 turns | 4 turns | 6 turns | 8 turns |
|---|---:|---:|---:|---:|
| Claude; GPT | −0.15* | −0.10* | +0.10* | +0.26* |
| Claude; Gemini | −0.34* | −0.38* | −0.33* | −0.23* |
| Claude; Qwen | −0.24* | −0.34* | −0.23* | −0.16* |
| GPT; Claude | +0.35* | +0.41* | +0.43* | +0.46* |
| GPT; Gemini | −0.05* | +0.01 | +0.02 | +0.05* |
| GPT; Qwen | +0.02 | +0.03* | +0.04* | +0.00 |
| Gemini; Claude | +0.44* | +0.59* | +0.44* | +0.43* |
| Gemini; GPT | +0.25* | +0.26* | +0.25* | +0.35* |
| Gemini; Qwen | +0.18* | +0.18* | +0.16* | +0.12* |
| Qwen; Claude | +0.42* | +0.47* | +0.55* | +0.54* |
| Qwen; GPT | +0.25* | +0.24* | +0.30* | +0.37* |
| Qwen; Gemini | +0.01 | +0.03 | +0.10* | +0.29* |

*Asterisks reproduce the first draft's `p<0.05` labels.*

The decision-level table also showed strong model dependence. GPT selected its
own conversation 82–100% of the time. Claude strongly preferred itself against
Gemini and Qwen but not against GPT. Gemini and Qwen usually preferred the
larger closed-source competitors.

| Judge; opponent | 2 turns | 4 turns | 6 turns | 8 turns |
|---|---:|---:|---:|---:|
| Claude; GPT | 38% | 27% | 33% | 46% |
| Claude; Gemini | 91%* | 93%* | 91%* | 93%* |
| Claude; Qwen | 90%* | 93%* | 95%* | 98%* |
| GPT; Claude | 82%* | 93%* | 96%* | 98%* |
| GPT; Gemini | 99%* | 100%* | 100%* | 100%* |
| GPT; Qwen | 100%* | 99%* | 100%* | 100%* |
| Gemini; Claude | 5% | 2% | 1% | 2% |
| Gemini; GPT | 1% | 1% | 2% | 7% |
| Gemini; Qwen | 55% | 54% | 61%* | 76%* |
| Qwen; Claude | 8% | 3% | 5% | 2% |
| Qwen; GPT | 5% | 2% | 2% | 0% |
| Qwen; Gemini | 52% | 58%* | 53% | 37% |

The expanded MedSP1000 results in Section 8 are methodologically different and
do not replicate a monotonic length effect. This distinction should replace the
first draft's broader claim that self-preference necessarily strengthens with
conversation length.

## 6. Development and patient-simulator pilots

### 6.1 ChatDoctor patient-simulator comparisons

Two 50-case paired generation pilots compared
`mistralai/Mistral-Small-3.1-24B-Instruct-2503` with
`Qwen/Qwen3.8-27B-FP8`. Both models completed all 150 requested patient turns
without empty replies in both pilots.

| Pilot | Simulator | Mean case latency | Median case latency | Mean reply length |
|---|---|---:|---:|---:|
| Opening-message grounded | Mistral | 0.265 s | 0.201 s | 207.8 characters |
| Opening-message grounded | Qwen 27B | 1.071 s | 0.795 s | 201.9 characters |
| Reference-note grounded | Mistral | 0.460 s | 0.344 s | 378.1 characters |
| Reference-note grounded | Qwen 27B | 1.070 s | 0.851 s | 234.9 characters |

The generated blinded-review CSVs each contain 50 cases, but every rating and
preference field is blank. Therefore, the pilots support reliability and speed
comparisons only; there is no completed human evidence that one simulator is
more natural, faithful, or preferable. The opening-message-only pilot also
cannot verify facts absent from the opening message. The reference-grounded
pilot gives the simulator a source doctor response, which is still not a
structured patient record.

### 6.2 MedSP1000 generation-only pilots

One Qwen-patient/Qwen-clinician case and one Mistral-patient/Qwen-clinician case
confirmed the four-exchange role-separated pipeline. A later deprecated v1
smoke exposed unsupported patient details and task drift, motivating the v2
prompt and strict information boundary. These are engineering and qualitative
prompt-development findings, not comparative clinical results.

## 7. Expanded Real-POCQi experiments

### 7.1 Eight-model generation and generation QA — complete

The append-only generation artifact contains 5,051 physical attempts: 5,047
successful rows and four failed Opus truncation attempts. After selecting the
latest success for each question-model cell, all 4,960 intended generations are
present. Median output lengths show a large verbosity difference:

| Generator | Median output tokens | 5th percentile | 95th percentile | Maximum |
|---|---:|---:|---:|---:|
| `claude-opus-5` | 2,927 | 1,886 | 4,658 | 8,045 |
| `claude-sonnet-5` | 1,354 | 964 | 1,937 | 5,915 |
| `Qwen/Qwen3.8-27B-FP8` | 988 | 577 | 1,613 | 2,187 |
| `Qwen/Qwen3.5-122B-A10B-FP8` | 875 | 579 | 1,303 | 1,837 |
| `gpt-5.6-terra` | 822 | 312 | 1,915 | 3,439 |
| `gpt-5.6-sol` | 720 | 284 | 1,753 | 4,519 |
| `gemini-3.7-flash` | 711 | 487 | 1,120 | 1,616 |
| `gemini-3.1-pro-preview` | 656 | 462 | 994 | 1,207 |

Manual review of three matched cases found a substantive venous-reflux error
in Sonnet, an internal HSV-trial contradiction in Sonnet, and omissions or
unsupported claims across models in an oncology trial question. The generation
set is therefore a valid evaluation input but not a clinical gold standard.
Opus verbosity is a plausible judging confound.

### 7.2 Blinded rubric plus explicit ranking — complete

The final logical matrix has 4,960 judgments: 620 questions × eight judges.
Every judgment ranks all eight blinded candidates and scores five rubric axes.
The aggregate generator results are:

| Generator | Mean rank | Ranked first | Mean rubric sum (of 25) |
|---|---:|---:|---:|
| `claude-opus-5` | 1.887 | 69.3% | 23.110 |
| `gpt-5.6-terra` | 3.371 | 17.5% | 21.652 |
| `gemini-3.7-flash` | 3.919 | 1.3% | 21.079 |
| `gpt-5.6-sol` | 4.301 | 8.3% | 20.838 |
| `claude-sonnet-5` | 4.389 | 1.5% | 20.632 |
| `gemini-3.1-pro-preview` | 4.620 | 1.3% | 20.414 |
| `Qwen/Qwen3.5-122B-A10B-FP8` | 6.524 | 0.4% | 17.524 |
| `Qwen/Qwen3.8-27B-FP8` | 6.989 | 0.3% | 16.406 |

Absolute score calibration differs substantially by judge, so raw rubric
scores should not be pooled without judge normalization. Explicit ranking and
rubric-sum ordering agree on 94.0% of candidate pairs but produce the identical
eight-model order in only 27.6% of judgments.

The matched, position-adjusted self-preference result is negative for every
judge:

| Judge | Raw own-answer mean rank | Adjusted self-preference | 95% CI |
|---|---:|---:|---:|
| `gpt-5.6-sol` | 1.735 | −2.939 | [−3.041, −2.836] |
| `gpt-5.6-terra` | 1.442 | −2.224 | [−2.325, −2.123] |
| `gemini-3.7-flash` | 2.844 | −1.236 | [−1.317, −1.155] |
| `claude-opus-5` | 1.040 | −0.969 | [−1.019, −0.919] |
| `Qwen/Qwen3.5-122B-A10B-FP8` | 5.884 | −0.730 | [−0.852, −0.608] |
| `gemini-3.1-pro-preview` | 4.023 | −0.683 | [−0.790, −0.575] |
| `claude-sonnet-5` | 3.816 | −0.647 | [−0.763, −0.530] |
| `Qwen/Qwen3.8-27B-FP8` | 6.682 | −0.324 | [−0.419, −0.228] |

The pooled effect is −1.219 positions (95% CI −1.261 to −1.177). This matched
analysis changes the interpretation of the Qwen judges. They rarely rank their
answers first because their answers are weak overall, but each still moves its
own answer upward relative to how the other judges rank the same answer.

The draft-compatible measures make this distinction explicit:

| Model/judge | Raw SP-Bias (95% CI), 0–5 points | Own-pick rate |
|---|---:|---:|
| `Qwen/Qwen3.5-122B-A10B-FP8` | +0.642 [+0.596, +0.688] | 30.2% |
| `Qwen/Qwen3.8-27B-FP8` | +0.002 [−0.047, +0.052] | 18.8% |
| `claude-opus-5` | +0.016 [+0.003, +0.028] | 99.4% |
| `claude-sonnet-5` | −0.091 [−0.116, −0.067] | 59.8% |
| `gemini-3.1-pro-preview` | +0.553 [+0.519, +0.587] | 56.8% |
| `gemini-3.7-flash` | +0.668 [+0.647, +0.689] | 73.7% |
| `gpt-5.6-sol` | +0.321 [+0.297, +0.345] | 89.5% |
| `gpt-5.6-terra` | −0.749 [−0.785, −0.712] | 93.7% |

For example, Terra is strongly self-critical on the raw absolute score scale
but ranks its own answer above 93.7% of competitors. Opus has almost no raw
score inflation while choosing its own, genuinely high-performing answer in
99.4% of derived head-to-head decisions. Qwen 122B shows strong raw score
inflation but only a 30.2% own-pick rate because its answers rank poorly in
absolute terms. This is why neither raw SP-Bias nor own-pick rate should be used
alone as the primary self-preference measure.

### 7.3 Qwen-judge extension — complete

The two Qwen judges contributed 1,240 completed judgments to the combined
matrix. Pooled over Qwen judges only, Opus remains first (mean rank 1.330),
followed by Terra (3.944), Gemini Flash (3.963), Sonnet (4.260), Gemini Pro
(4.706), Sol (5.031), Qwen 122B (6.133), and Qwen 27B (6.634). Qwen judges use
the top of the absolute score scale often, assigning exactly 5 on 62.5% and
48.1% of axis scores for the 122B and 27B judges, respectively. This supports
the broad generator ordering while reinforcing the need for judge-specific
score calibration.

### 7.4 No-rubric direct ranking — complete

The direct condition contains 800 completed judgments on a deterministic
100-question sample. Judges saw no explicit rubric and ranked the eight blinded
answers by overall quality.

| Generator | Mean rank | Ranked first |
|---|---:|---:|
| `claude-opus-5` | 1.834 | 72.4% |
| `gpt-5.6-terra` | 3.357 | 15.4% |
| `gemini-3.7-flash` | 3.830 | 0.6% |
| `claude-sonnet-5` | 4.350 | 1.0% |
| `gpt-5.6-sol` | 4.350 | 10.0% |
| `gemini-3.1-pro-preview` | 4.610 | 0.5% |
| `Qwen/Qwen3.5-122B-A10B-FP8` | 6.574 | 0.1% |
| `Qwen/Qwen3.8-27B-FP8` | 7.095 | 0.0% |

On the same judge-question pairs, direct and combined explicit rankings agree
on 91.5% of candidate pairs, choose the same first-place answer 87.1% of the
time, and produce exactly the same eight-answer order 16.1% of the time. Thus,
removing the rubric changes fine ordering but not the dominant performance
pattern. Matched self-preference also remains strong in the direct condition;
seven judge estimates are negative, while the Qwen 27B interval includes zero.

### 7.5 Identity-revealed ranking — complete

The experiment reveals generator names for the same eight responses on a
seed-42 sample of 200 questions, using the six API judges. The final append-only
artifact contains all 1,200 planned logical judgments. The comparison below
restricts the blinded condition to the identical 200 questions and six judges,
then fits presentation-position effects separately in each condition.

Matched and position-adjusted self-preference is −1.391 positions in the
revealed condition versus −1.514 for the same question-judge cells when
blinded. The paired revealed-minus-blinded change is +0.123 positions (95% CI
+0.059 to +0.188; nominal `p=0.00019`), meaning that self-preference is modestly
weaker when names are shown.

| Judge | Revealed effect | Matched blinded effect | Revealed minus blinded |
|---|---:|---:|---:|
| `claude-opus-5` | −1.153 | −1.256 | +0.104 |
| `claude-sonnet-5` | −0.795 | −0.832 | +0.038 |
| `gemini-3.1-pro-preview` | −0.449 | −0.749 | +0.300 |
| `gemini-3.7-flash` | −1.177 | −1.315 | +0.138 |
| `gpt-5.6-sol` | −2.742 | −2.826 | +0.083 |
| `gpt-5.6-terra` | −2.028 | −2.105 | +0.077 |
| **Pooled within question** | **−1.391** | **−1.514** | **+0.123** |

Every revealed-condition effect remains negative and its 95% interval excludes
zero. After Holm correction across the six judge-specific changes, the
reductions for Claude Opus (`+0.104`) and Gemini Pro (`+0.300`) remain below
0.05; the changes for the other four judges do not. Explicit labels therefore
modestly reduce the pooled effect but do not eliminate same-model affinity.

## 8. Expanded MedSP1000 multi-turn experiments

### 8.1 Six-model trajectory generation — complete

The append-only generation artifact has 1,261 rows: 1,217 successful generation
keys and 44 failed attempts (39 runtime errors and five overloaded-provider
errors). Restricting to the production four-exchange, medium-reasoning
configuration yields exactly 1,200 question-model cells: 200 for each of the
six clinician models. The Mistral patient simulator is fixed across models.

Median total clinician output over four exchanges is 936 tokens for Sonnet,
706 for Opus, 658 for Sol, 537 for Terra, 319 for Gemini Pro, and 303 for Gemini
Flash. This difference may affect judgments and should be modeled as a possible
verbosity confound.

### 8.2 Combined rubric plus ranking by transcript length

The same six completed trajectories per question are truncated to 2, 4, or 6
visible turns or shown in full at 8 turns. All four judging matrices are
complete. Early batches of the same combined condition are retained in the
8-turn artifact as part of the completed run.

| Visible turns | Latest successes | Complete six-judge questions | Status | Pooled self-preference (95% CI) |
|---:|---:|---:|---|---:|
| 2 | 1,200/1,200 | 200/200 | Complete | −0.518 [−0.585, −0.452] |
| 4 | 1,200/1,200 | 200/200 | Complete | −0.457 [−0.527, −0.388] |
| 6 | 1,200/1,200 | 200/200 | Complete | −0.502 [−0.574, −0.430] |
| 8 | 1,200/1,200 | 200/200 | Complete | −0.613 [−0.683, −0.543] |

The corresponding effects are −0.518, −0.457, −0.502, and −0.613. The
four-length paired analysis uses all 200 questions; the 8-minus-2 change is
−0.094 positions (95% CI −0.187 to −0.002). The sequence is
not monotonic.

Judge-specific effects also vary:

| Judge | 2 turns | 4 turns | 6 turns | 8 turns |
|---|---:|---:|---:|---:|
| `claude-opus-5` | −0.153 | −0.218 | −0.423 | −0.445 |
| `claude-sonnet-5` | −0.233 | −0.378 | −0.549 | −0.472 |
| `gemini-3.1-pro-preview` | −0.791 | −0.554 | −0.435 | −0.545 |
| `gemini-3.7-flash` | −0.572 | −0.600 | −0.428 | −0.645 |
| `gpt-5.6-sol` | −0.536 | −0.487 | −0.539 | −0.938 |
| `gpt-5.6-terra` | −0.826 | −0.508 | −0.637 | −0.632 |

Draft-compatible raw SP-Bias and own-pick rates also disagree across models and
lengths. Each cell below is `SP-Bias / own-pick rate`:

| Model/judge | 2 turns | 4 turns | 6 turns | 8 turns |
|---|---:|---:|---:|---:|
| `claude-opus-5` | −0.517 / 54.1% | −0.267 / 56.8% | −0.159 / 67.6% | −0.086 / 85.5% |
| `claude-sonnet-5` | −0.313 / 53.8% | −0.020 / 76.8% | +0.036 / 82.6% | −0.015 / 77.4% |
| `gemini-3.1-pro-preview` | +0.831 / 64.1% | +0.624 / 47.5% | +0.468 / 39.0% | +0.321 / 33.0% |
| `gemini-3.7-flash` | +0.842 / 51.4% | +0.711 / 49.5% | +0.705 / 43.7% | +0.764 / 44.9% |
| `gpt-5.6-sol` | +0.244 / 67.2% | +0.113 / 62.6% | +0.156 / 63.6% | +0.278 / 72.0% |
| `gpt-5.6-terra` | −0.322 / 61.1% | −0.257 / 52.5% | −0.094 / 53.8% | +0.110 / 48.7% |

The Gemini judges illustrate the quality confound in the decision measure:
they often inflate scores for their own conversations while own-pick rates fall
as their generators lose relative rank at longer lengths. Conversely, Opus's
own-pick rate rises as its full conversations become stronger even though its
raw SP-Bias approaches zero.

The strongest single effect is Sol at eight turns. Opus and Sonnet generally
become more self-favorable as more conversation is visible, whereas Gemini Pro
and Terra begin with relatively large 2-turn effects and do not increase
monotonically. This judge heterogeneity is more informative than a single
pooled “length increases bias” claim.

Aggregate clinician performance also changes with length:

| Clinician generator | Mean rank, 2 turns | 4 turns | 6 turns | 8 turns |
|---|---:|---:|---:|---:|
| `claude-opus-5` | 3.417 | 3.335 | 2.968 | 2.097 |
| `claude-sonnet-5` | 3.505 | 2.478 | 2.333 | 2.527 |
| `gpt-5.6-sol` | 3.083 | 3.268 | 3.264 | 3.180 |
| `gpt-5.6-terra` | 3.634 | 3.803 | 3.847 | 4.098 |
| `gemini-3.7-flash` | 3.911 | 4.032 | 4.178 | 4.298 |
| `gemini-3.1-pro-preview` | 3.451 | 4.084 | 4.410 | 4.800 |

Sol ranks first at two turns, Sonnet at four and six turns, and Opus at eight
turns. Opus improves markedly as the full interaction becomes visible, while
both Gemini models lose relative rank. This could reflect better longitudinal
history-taking by Anthropic models, verbosity, adaptation across turns, or
rubric sensitivity; it is not itself evidence of self-preference.

### 8.3 Token-length sensitivity

We measured the exact candidate text with a common lexical tokenizer rather
than comparing provider usage tokens across incompatible tokenizers. For
MedSP1000, the primary length is the clinician text within each visible prefix.
A fixed-effects regression controlling the question–judge candidate list,
generator model, and presentation position finds that a doubling of length is
associated with a 1.314-position better Real-POCQi rank (95% CI 1.202 to 1.426)
and 1.351, 2.084, 2.353, and 2.279-position better MedSP1000 ranks at 2, 4, 6,
and 8 turns. Normalized rubric scores show the corresponding positive
associations. These are observational associations: clinical completeness and
quality may cause both length and score, so the coefficients should not be
interpreted as a causal verbosity bonus.

The primary self-preference estimand already holds the exact answer—and hence
its length—fixed between the own judge and outside judges. Length therefore
cannot be an omitted answer-level confounder of that matched contrast. It does
moderate the effect: per length doubling, the matched rank effect changes by
+0.538 positions in Real-POCQi (weaker self-preference) but by −0.259, −0.306,
−0.334, and −0.274 positions across the four MedSP1000 prefixes (stronger
self-preference). Provider-token sensitivity analyses preserve the directions.
Thus token length is important for absolute model scoring and heterogeneous
self-preference, but it does not provide a common explanation for the matched
self-preference observed in both tasks.

## 9. Cross-experiment synthesis

Several conclusions are supported across the current artifacts:

1. **Matched self-preference is widespread.** Every judge in the complete
   blinded Real-POCQi condition and every six-model MedSP1000 condition has a
   negative average matched rank difference.
2. **Judge identity matters more than task subgroup.** Real-POCQi effects range
   from −0.324 to −2.939 across judges, while specialty heterogeneity is not
   significant.
3. **Raw wins are an incomplete measure.** Qwen answers rarely win but are
   still promoted by their own judges relative to outside judges. Conversely,
   Opus has a near-perfect raw own rank partly because other judges also prefer
   its answers.
4. **Rubric removal does not remove the main pattern.** Direct and combined
   Real-POCQi rankings agree on 91.5% of candidate pairs.
5. **Length matters for scoring but does not explain away matched preference.**
   Longer answers score better conditionally, yet the matched design holds the
   exact answer fixed and length moderation has opposite signs across tasks.
6. **Longer interaction effects are real but not universal.** Eight-turn
   MedSP1000 self-preference is larger than at two turns, but the 2→4→6→8 trend
   is not monotonic and varies by judge.
7. **Identity visibility does not amplify preference.** The completed revealed
   experiment instead shows a modest reduction.
8. **Model-family affinity is plausible.** In Real-POCQi, the pooled same-family
   preference (−1.269 positions) is close to the exact-model effect (−1.219).
   Because identities were hidden in the primary condition, this can arise from
   shared style, reasoning conventions, or evaluation criteria rather than
   explicit self-recognition.

## 10. Limitations and next analyses

- The legacy results cannot be independently reproduced from this repository
  because their raw generations and judgments are absent.
- The expanded conditions use one generation per model-question cell; repeated
  generations are needed to distinguish stable family affinity from answer-
  specific idiosyncrasy.
- Judge calibration differs markedly, especially for Qwen, so absolute scores
  require within-judge normalization or hierarchical judge effects.
- Candidate position affects rankings. Randomization limits systematic model
  confounding, but position should remain a covariate.
- Response length differs greatly by model and is strongly associated with
  rankings. The current sensitivity is observational and cannot separate a
  verbosity reward from genuine completeness or answer quality.
- Specialty strata range from 3 to 55 questions, leaving many subgroup analyses
  underpowered.
- The identity-revealed comparison contains six API judges rather than the two
  Qwen judges, so its disclosure effect should not be generalized to every
  model in the blinded cohort.
- The current MedSP1000 scientific matrix has six API models, not the planned
  eight-model cohort. Qwen trajectory generation and judging remain future work.
- Automated judges have not yet been benchmarked against blinded physician
  judgments in these expanded conditions.

The strongest next extension is a hierarchical rank model with question,
generator, judge, model family, position, response length, specialty, identity
disclosure, and transcript-length interactions. MedSP1000 should add the two
Qwen generators and judges before making claims about open-weight models in the
multi-turn setting.

## 11. Conclusion

The project now contains much stronger evidence than the original four-model
draft. In a complete eight-model, 620-question blinded Real-POCQi study, every
judge ranks its own answer more favorably than other judges rank that same
answer, even when the answer rarely wins. The effect is driven much more by the
judge than by clinical specialty. In six-model MedSP1000 conversations,
self-preference persists from two through eight turns but does not increase in
a simple monotonic way. Revealing generator names modestly reduces rather than
amplifies the matched effect, although same-model affinity remains strong.
These results favor multi-family judge panels, matched estimands, explicit
position controls, and separate reporting of scores and rankings over reliance
on a single LLM judge or raw own-win rate.

---

# Appendix A. Complete specialty-specific analysis

This appendix incorporates the findings from
`docs/real_pocqi_self_preference_by_specialty.md` into the updated draft.

## A.1 Specialty estimand and overall result

The analysis uses the latest successful record for each logical judge-question
key in the complete blinded Real-POCQi combined condition. The primary outcome
is the position-adjusted matched exact-model rank difference defined in Section
4.2. Pooled intervals average the eight correlated judge observations within
each question before estimating uncertainty.

Across all judges and questions, the adjusted effect is −1.219 positions (95%
CI −1.261 to −1.177). The corresponding same-family effect is −1.269 (95% CI
−1.311 to −1.227). First-place uplift is not the primary outcome because it
misses within-list movement, particularly for weak generators.

## A.2 Specialty estimates

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
| Urology | 8 | −1.119 | — | −0.897 | −1.108 |
| Hospital-Based Medicine | 7 | −0.862 | — | −0.684 | −0.951 |
| Surgical Oncology | 6 | −1.063 | — | −0.913 | −1.136 |
| Genetics | 4 | −0.684 | — | −0.645 | −0.872 |
| Otolaryngology | 4 | −0.933 | — | −0.799 | −0.869 |
| Ophthalmology | 3 | −1.056 | — | −0.997 | −1.301 |

Intervals are omitted for strata with fewer than 10 questions because they are
too sparse for useful comparison. Their point estimates remain descriptive.

## A.3 Specialty heterogeneity tests

A label-permutation test preserving the 30 observed specialty sizes found that
specialty explains 5.7% of question-level variation, with `p=0.190` over 10,000
permutations. The observed range—from −1.543 in Family Medicine to −0.908 in
Radiology among specialties with at least 10 questions—is therefore compatible
with sampling variation across many uneven groups.

No judge-specific specialty test crosses 0.05:

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

Ranking candidates by their five-axis rubric sums yields the same effect
direction for every judge. Across 30 specialties, the explicit-rank and rubric-
sum-rank effects correlate at `r=0.889`. Presentation-order adjustment makes
almost no difference to the overall estimate. The specialty conclusion is
therefore robust across the two ranking constructions and order adjustment.

The correct interpretation is not that every specialty has an identical true
effect. Rather, this dataset supports a general self-preference effect and does
not have convincing evidence for a specialty interaction. A confirmatory
subspecialty analysis should predefine fewer groups, increase the small strata,
and use a hierarchical rank model.

---

# Appendix B. Artifact map and status rules

| Purpose | Artifact |
|---|---|
| Frozen Real-POCQi questions | `data/question_sets/real_pocqi_questions.jsonl` |
| Real-POCQi generations | `data/outputs/generations/real_pocqi_generations.jsonl` |
| Complete blinded combined judgments | `data/real_pcoqi/judgements/rubric_and_model_ranking.jsonl` |
| Complete direct rankings | `data/real_pcoqi/judgements/direct_ranking.jsonl` |
| Complete identity-revealed judgments | `data/real_pcoqi/judgements/identity_revealed_rubric_and_model_ranking.jsonl` |
| Frozen MedSP1000 questions | `data/question_sets/medsp1000_generation_cases.jsonl` |
| MedSP1000 trajectories | `data/outputs/medsp1000/generations.jsonl` |
| Complete 2-turn judgments | `data/outputs/medsp1000/judgements/rubric_and_model_ranking_2_turns.jsonl` |
| Complete 4-turn judgments | `data/outputs/medsp1000/judgements/rubric_and_model_ranking_4_turns.jsonl` |
| Complete 6-turn judgments | `data/outputs/medsp1000/judgements/rubric_and_model_ranking_6_turns.jsonl` |
| Complete 8-turn judgments | `data/outputs/medsp1000/judgements/rubric_and_model_ranking.jsonl` |
| Patient-simulator pilot summaries | `data/outputs/pilots/*/summary.json` |

All production JSONL files are append-only and retain failures, retries, smoke
records, and superseded successful attempts. Analyses must filter by
`experiment_id` and `status`, then select the latest successful record for each
logical key. Physical line counts are not observation counts.

## References

Alvarez-Arenas, J. I., D. Jimenez-Carretero, D. Mananes, and F. Sanchez-Cabo.
“The Unreliable Judges: Assessing Reproducibility and Self-Preference Bias of
LLMs as Free-Text Evaluators.” *medRxiv*, 2026.

Li, Lingyao, et al. “LLM-as-a-Judge in Healthcare: A Scoping Analysis of
Applications, Methods, and Human Alignment.” *arXiv:2605.25273*, 2026.

Panickssery, Arjun, Samuel R. Bowman, and Shi Feng. “LLM Evaluators Recognize
and Favor Their Own Generations.” *NeurIPS 37*, 2024.

Pombal, Jose, Ricardo Rei, and Andre F. T. Martins. “Self-Preference Bias in
Rubric-Based Evaluation of Large Language Models.” *arXiv:2604.06996*, 2026.

Wataoka, Koki, Tsubasa Takahashi, and Ryokan Ri. “Self-Preference Bias in
LLM-as-a-Judge.” *arXiv:2410.21819*, 2025.

Zheng, Lianmin, et al. “Judging LLM-as-a-Judge with MT-Bench and Chatbot
Arena.” *NeurIPS 36*, 2023.
