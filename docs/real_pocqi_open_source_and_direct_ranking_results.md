# Real-POCQi open-source judges and blind-ranking subset

QA performed on August 24, 2026 after extending the combined Real-POCQi
experiment to the two open-source Qwen judges and running a no-rubric ranking
condition on a deterministic random sample of 100 questions.

## Run design

- The combined condition covers all 620 questions and all eight candidate
  responses. It now has eight judges: the six API judges from the original run
  plus `Qwen/Qwen3.5-122B-A10B-FP8` and `Qwen/Qwen3.8-27B-FP8` on Modal.
- The direct-ranking condition uses the same eight judges on 100 questions
  selected with seed `20260824`.
- Direct-ranking judges received no predefined rubric. They were asked only to
  rank every blinded candidate from highest to lowest overall quality.
- Both conditions hide generator identities behind `response-1` through
  `response-8`. Generator provenance remains in the saved record so rankings
  can be resolved for analysis.
- Combined experiment: `real_pocqi_combined_all_judges_v1` in
  `data/real_pcoqi/judgements/rubric_and_model_ranking.jsonl`.
- Direct-ranking experiment: `real_pocqi_direct_ranking_random100_v1` in
  `data/real_pcoqi/judgements/direct_ranking.jsonl`.

## Execution integrity

The final artifacts passed all of the following checks:

- Combined: 4,960/4,960 unique logical judgments succeeded, exactly 620 from
  each of eight judges, with zero unresolved keys.
- Direct ranking: 800/800 unique logical judgments succeeded, exactly 100 from
  each judge, with zero unresolved keys.
- Every judge saw the same question set within each condition. The 100 direct
  questions are a subset of the 620 combined questions.
- Every JSONL line parses through `PocqiJudgmentRecord`; every successful
  ranking is a permutation of all eight candidates and every combined score is
  within the required 0--5 range.
- All final records have `identity_blinded=true`, and none of the eight
  generator-model names occurs in any judge-facing system or user prompt.
- Resume dry runs report 0 pending and skip all 4,960 combined and 800 direct
  logical judgments.
- The automated test suite passes with 77 tests.

The files are append-only. The combined experiment contains 5,476 physical
attempt records for 4,960 logical keys, and the direct experiment contains 816
attempt records for 800 logical keys. The additional attempts preserve audit
history from retries and deliberately interrupted resumability checks. Analysis
must select the latest successful record per judgment key.

## Combined results with all eight judges

Lower mean rank is better. Rubric sums range from 0 to 25.

| Generator | Mean rank | Ranked first | Mean rubric sum |
|---|---:|---:|---:|
| `claude-opus-5` | 1.887 | 69.3% | 23.110 |
| `gpt-5.6-terra` | 3.371 | 17.5% | 21.652 |
| `gemini-3.7-flash` | 3.919 | 1.3% | 21.079 |
| `gpt-5.6-sol` | 4.301 | 8.3% | 20.838 |
| `claude-sonnet-5` | 4.389 | 1.5% | 20.632 |
| `gemini-3.1-pro-preview` | 4.620 | 1.3% | 20.414 |
| `Qwen/Qwen3.5-122B-A10B-FP8` | 6.524 | 0.4% | 17.524 |
| `Qwen/Qwen3.8-27B-FP8` | 6.989 | 0.3% | 16.406 |

## Results from the Qwen judges alone

The two open-source judges preserve the broad aggregate ordering. Their pooled
rankings place Opus first by a wide margin and both Qwen generations last.

| Generator | Mean rank | Ranked first | Mean rubric sum |
|---|---:|---:|---:|
| `claude-opus-5` | 1.330 | 80.0% | 24.865 |
| `gpt-5.6-terra` | 3.944 | 6.0% | 23.110 |
| `gemini-3.7-flash` | 3.963 | 2.5% | 22.666 |
| `claude-sonnet-5` | 4.260 | 4.6% | 22.377 |
| `gemini-3.1-pro-preview` | 4.706 | 2.7% | 21.852 |
| `gpt-5.6-sol` | 5.031 | 1.7% | 22.034 |
| `Qwen/Qwen3.5-122B-A10B-FP8` | 6.133 | 1.4% | 18.980 |
| `Qwen/Qwen3.8-27B-FP8` | 6.634 | 1.1% | 17.587 |

The Qwen judges use the upper end of the absolute scoring scale frequently:
the 122B judge has a mean axis score of 4.453 and assigns exactly 5 on 62.5% of
axis decisions; the 27B judge has a mean of 4.220 and assigns exactly 5 on
48.1%. Their explicit rankings agree with their own rubric-sum order on 90.3%
and 92.0% of candidate pairs, respectively.

Neither judge shows strong self-preference. The 122B judge gives its own
response a mean rank of 5.884 and ranks it first on 2.6% of questions. The 27B
judge gives its own response a mean rank of 6.682 and never ranks it first.

## No-rubric direct ranking on the random 100

Pooled over all eight judges, the direct condition produces the following
ordering:

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

For the same 100 judge-question pairs, direct rankings and the explicit
rankings from the combined rubric condition agree on 91.5% of all candidate
pairs. They select the same first-place response 87.1% of the time and produce
the exact same eight-response order 16.1% of the time. This indicates that
removing the explicit rubric changes some within-list ordering but not the main
performance pattern.

## Assessment

Both new artifacts are complete, correctly blinded, and ready for analysis.
The open-source judges broadly corroborate the earlier ordering and do not show
the strong family self-preference observed for some API judges. The direct
no-rubric condition is also highly consistent with the rubric-plus-ranking
condition, though it should remain a separate experimental condition rather
than being pooled as if the prompts were identical.
