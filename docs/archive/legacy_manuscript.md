# Med-Self-Preference first draft (legacy manuscript)

> Archived for provenance. The current experiment record, corrected claims,
> and expanded results are in [`../experiment.md`](../experiment.md). The raw
> generations and judgments underlying this legacy four-model draft are not
> present in this repository.

Zara Ansari<sup>1,2</sup>, Shlok Natarajan<sup>1</sup>, Aaron Fanous<sup>1</sup>, Roxana Daneshjou<sup>1,3</sup>

<sup>1</sup> Department of Biomedical Data Science, <sup>3</sup> Department of Dermatology, Stanford University, 450 Jane Stanford Way, Stanford, CA 94305, USA.<br>
<sup>2</sup> Department of Computer Science, Harvey Mudd College, 301 Platt Boulevard, Claremont, CA 91711, USA.

June 2026

## Abstract

Benchmarking medical AI has required increasingly larger datasets, making human evaluation untenable. The common approach now is to test AI models on a set of tasks and to use Large language models (LLMs) as automated judges to score and compare model performance. However, LLM judges may show self-preference bias, where a model favors responses produced by its own model. This has serious implications in medicine since LLM-as-a-judge is readily used and biased evaluations could affect which models are considered safe. This study examines LLM-as-a-judge self-preference with GPT-5.5, Claude Sonnet 5, Gemini 3.1 Flash-Lite, and Qwen 3.6 (35B) in two medical settings. We evaluate single-turn specialist question answering using a subset of the Real-POCQi dataset and multi-turn simulated physician-patient conversations using a subset of the HealthCareMagic dataset at lengths ranging from two to eight turns. When we measure self-preference model identities are concealed from the judge (identity-blind) and response order is randomized to control for positional bias. We assess self preference in two ways. Score-level bias compares how a model's own judge and an outside judge score the same response, allowing us to isolate differences caused by the judge rather than differences in answer quality. Decision-level self-preference measures how often a judge selects its own model's response in a head to head comparison without looking at the score. Our results show that self-preference is statistically significant but varies across models. Gemini and Qwen scored their own responses higher than external judges scored the same responses, but often selected competing responses in direct comparisons. In contrast, GPT-5.5 showed limited score inflation but selected its own response in 96-99% of comparisons. Claude was generally self-critical and often scored its own responses less favorably than outside judges. Additionally, both forms of self-preference became stronger as conversation length increased. These findings show that score level and decision level measures capture different forms of self preference bias and should be considered when evaluating LLM judges in medicine.

## 1 Introduction

The complexity of medical practice has called for larger and more complex benchmarking for comparing the performance of AI models. Human review of thousands of scenarios tested across multiple AI models is untenable due to cost and time [Zheng et al., 2023]. A recent review of medical AI benchmarking found that large language models (LLMs) are increasingly used as judges to evaluate the outputs of other LLMs (and in some cases, their own outputs) [Li et al., 2026]. This same study found that OpenAI's models are currently the most popular judges for medical AI benchmarking studies [Li et al., 2026].

Researchers claim using an LLM as a judge provides a scalable way to evaluate responses based on certain criteria such as accuracy, completeness, safety, and clarity. The stakes in medical AI evaluation are high, as biased judges could incorrectly favor unsafe models and influence clinical deployment decisions to the detriment of patients. Furthermore, evaluating medical responses is especially complicated since an answer may sound confidently correct, but still contain incomplete medical information. As a result, evaluating LLMs used in the medical context is a multifaceted process where faithfulness, completeness, safety, clarity, and conciseness should be considered.

Prior work in general generative AI tasks has shown that LLM judges may exhibit biases such as positional and verbosity bias. Wataoka et al. [2025] examined how LLM judges may favor responses that reflect policies or writing styles familiar to them. In fact, one study showed that LLM judges may favor AI generated clinical responses over human produced ones. Alvarez-Arenas et al. [2026] found that this preference seemed to depend more on how an answer was written, especially how long it was and the language used, than on whether the model could actually tell if the answer came from a human or an AI. In one experiment, long responses still received high scores even when matched with unrelated questions, demonstrating that verbosity can outweigh the content for some LLM judges. Wataoka et al. [2025] introduced a quantitative way to measure self-preference in LLM judges. They found that GPT-4 showed significant self-preference and that LLM judges tended to rate lower perplexity responses more highly than human evaluators did. Work from Pombal et al. [2026] looked at whether rubric-based evaluations (with a set of yes or no criteria) could mitigate the self-preference of LLM judges. Their work found that LLM self-preference persists even with fully "objective" rubrics.

The literature for self-preference in medical contexts is far more sparse. To address these gaps, we evaluate self-preference across four model families in both single-turn generalist-specialist question answering and multi-turn patient-physician conversations. In order to address the fact that a model may choose its own response because its answer is genuinely better, we evaluate the models preference in two distinct ways. We use both score-level self-preference, which quantifies how different judges score the same fixed response, and decision-level self-preference, which measures how often a judge selects its own model's response in a head to head comparison. Our findings of LLM judge self-preference in medical contexts provide important considerations for future medical AI benchmarking studies.

## 2 Methods

In this study we use subsets of two different different datasets whose data are described here. The Real-POCQi dataset contains 620 real world clinical questions which were submitted by practicing U.S. physicians through the OpenEvidence platform. The questions span 30 medical specialties and focus on point-of-care clinical decision making. Each entry includes a unique question ID, the question text, and its respective specialty. The dataset was reviewed to remove identifying patient information and other content that could raise privacy concerns. In this study, we use these questions as prompts for generating and evaluating responses from the four tested models.

The ChatDoctor-HealthCareMagic-100k dataset contains 112,165 patient-physician question and answer pairs collected from HealthCareMagic, which is an online medical consultation platform. Each entry includes a patient's description of their symptoms or medical concern, along with a corresponding response written in the style of a physician. The dataset also includes instructions asking the model to answer the medical question based on the patient's description. In this study, we use the patient questions as the starting point for generating multi-turn patient-physician conversations across the four models.

### 2.1 Sample Size Analysis

We conducted a power analysis using a two-sided significance level of 0.05 and 90% power. For the score-level analysis (explained in the following sections), about 119 paired observations were needed to detect a small-to-medium effect ((d=0.30)). For the decision-level analysis, about 114 observations were needed to detect an own-pick rate of 65%, compared with the 50% rate expected under no preference. Based on these estimates, using 125 scenarios gave us enough power for both analyses while still leaving some room for missing or unusable data.

In the single-turn setting, each of the four models answered all 125 sampled questions, resulting in 500 specialist responses (125 questions x 4 models). These responses were then evaluated pairwise. This means that across the four models there are six unique model pairs (C(4,2) = 6), and within each pair both models served as judges (2 judges per pair), each judging all 125 questions. This yields 1,500 pairwise judgments (6 pairs x 2 judges x 125 questions), where each judgment consists of one judge scoring both conversations and selecting a preferred response (the scoring and the preferred response selection are independent). In the multi-turn setting, each of the four models generated an eight-turn conversation for all 125 scenarios, producing 500 conversations (125 scenarios x 4 models); the shorter 2, 4, and 6 turn versions were derived by truncating these eight turn conversations. Because pairwise judging was repeated at each of the four conversation lengths, the same six pairs and two judges per pair were applied four times, giving 6,000 pairwise judgments (6 pairs x 2 judges x 4 lengths x 125 scenarios). In total, our analysis is based on 7,500 identity-blind, order-randomized pairwise judgments across the two settings.

### 2.2 Generator and Judge Models

In this study, we have four generator models that we use in both the single turn and multi turn pipelines. Generator models are fed the original query from either the Real-POCQi or ChatDoctor-HealthCareMagic-100k dataset and produce an answer in response to that given query. We use GPT-5.5, Claude Sonnet 5, Gemini 3.1-flash-lite, and Qwen 3.6:35b (run locally via Ollama). These same models were used as judges (refer to Table 1).

### 2.3 Single-turn Pipeline

This pipeline evaluates specialist model behavior on single turn questions using 125 questions sampled from the jjfenglab/Real-POCQi dataset, where each question carries a clinical specialty (e.g., cardiology, neurology). For each question, four models (GPT-5.5, Claude Sonnet 5, Gemini 3.1-flash-lite, and Qwen 3.6:35b) each independently produce a single answer in the role of the relevant specialist, prompted with that specialty's label. Each model generates exactly one response per question.

These responses are evaluated in a pairwise (head to head) manner across five criteria that are inspired by MedHELM style clinical evaluation rubrics: faithfulness, completeness, safety, clarity, and conciseness (0-5; overall = their mean). For each of the six model pairings, both models act as judges (as shown in Table 1). They are identity-blind, with A/B order randomized per comparison under a fixed seed so both judges see the same ordering, and in one pass each judge scores both answers on the five criteria and selects a preferred one (A, B, or tie) using its own reasoning. The pairwise judgments yield, for every comparison, an overall score for each answer and a preferred answer (the preferred answer is not based on the score, but on the model's judgment). These feed the two measures of self-preference: score-level and decision-level.

**Table 1. Generator pairs and corresponding judge models. Each judge evaluates the pair in a separate run.**

| Generator pair | Judges |
|---|---|
| Claude vs. GPT | Claude, GPT (separate judge runs) |
| Claude vs. Gemini | Claude, Gemini (separate judge runs) |
| Claude vs. Qwen | Claude, Qwen (separate judge runs) |
| GPT vs. Gemini | GPT, Gemini (separate judge runs) |
| GPT vs. Qwen | GPT, Qwen (separate judge runs) |
| Gemini vs. Qwen | Gemini, Qwen (separate judge runs) |

### 2.4 Multi-turn Pipeline

This pipeline evaluates physician model behavior across multi-turn patient-physician medical conversations using 125 scenarios randomly sampled from the lavita/ChatDoctor-HealthCareMagic-100k dataset. For each of the 125 selected scenarios from the dataset, we used the original patient query as the first patient message in the simulated conversation. Each of four physician models, GPT-5.5, Claude Sonnet 5, Gemini 3.1-flash-lite, and Qwen 3.6:35b independently generates an 8 turn conversation for every scenario, alternating physician turns (the model under study) with patient turns supplied by a standardized patient simulator (GPT-5.5). Shorter conversation lengths (2, 4, and 6 turns) are derived by truncating each 8 turn conversation into the respective turn length. Since every turn only needs the preceding history conversation, truncating the 8 turn conversation into different lengths is a valid shorter conversation.

Similar to the single-turn pipeline, conversations in the multi-turn pipeline are evaluated in a pairwise manner across the same five criteria (0-5; overall = their mean). For each of the six model pairs, both models act as judges. They are identity-blind, with A/B order randomized per comparison under a fixed seed so both judges see the same ordering, and in one pass each judge scores both conversations (for a given turn length) and selects a preferred one (A, B, or tie) using its own reasoning. The difference from the single turn pipeline is that this pairwise judging is performed separately at each of the four conversation lengths (2, 4, 6, and 8 turns). Thus, the pairwise judgments, and the two resulting measures of self-preference, are computed and reported per model pair and per conversation length.

## 3 Analysis of Data

We use pairwise evaluation, where two AI generator models (e.g, Claude and GPT) answer the same question for a given dataset, dependent on whether it is the single turn or multi turn pipeline. Following this, responses are matched by their shared scenario-id, so that both generated answers correspond to the same medical case. The original question along with the two generated answers are presented side by side to an LLM judge as Response A and Response B. The order of the two responses is randomized independently for each scenario so that a given generator model is not consistently presented first.

The evaluation criteria were adapted from the MedHELM rubric to capture dimensions relevant to the quality of medical AI responses. For each response, the judge assigns a score from 0 to 5 on five criteria:

- **Faithfulness:** Medical accuracy and appropriateness.
- **Completeness:** Whether the response addresses the patient's concerns and provides appropriate guidance and follow-up.
- **Safety:** Whether the response avoids harmful advice and identifies relevant warning signs or emergency concerns.
- **Clarity:** Whether the response is understandable to a patient.
- **Conciseness:** Whether the response has an appropriate level of detail without unnecessary repetition.

After scoring both responses, the judge makes an overall pairwise preference decision: Response A is better, Response B is better, or tied (the responses are approximately equivalent).

### 3.1 Score Level Self Preference (SP-Bias)

For a given model, we hold its own conversation (or single answer in the single turn cases) fixed and take the difference between two judges' overall scores of it: the model's own-family judge's overall minus the other judge's overall, averaged across scenarios. Because both judges evaluate the same response, differences in response quality are held constant, allowing us to isolate differences in how the judges score it. A positive value means a model's own judge scores its conversations more generously than an outside judge does, while a negative score implies that the other model is scored better.

### 3.2 Decision Level Self Preference

Using each judge's preferred-conversation selection, we measure how often a judge picks its own family's conversation over the other model's. Unlike SP-Bias, this reflects the judge's holistic head-to-head choice rather than its numeric scores (the two conversations genuinely differ, so a stronger conversation can win on merit). The numerical scores (for SP-bias) and the head to head preference decision were made independently; the scores did not determine or influence which response the judge selected as preferred.

### 3.3 Statistical Analysis

For the score-level self-preference analysis, we used a paired one-sample t-test because the same answer was scored by two different judges. One score came from the model's own-family judge and the other came from the outside judge. Since both judges scored the exact same answer, the scores are paired. This allowed us to focus on differences caused by the judge rather than differences in answer quality. For each scenario, we calculated the difference between the two overall scores and tested whether the average difference was significantly different from zero. A positive difference means the model's own judge scored its answer more highly. A negative difference means the outside judge gave the higher score. We also report the p-value and 95% confidence interval for each comparison.

For the decision-level preference, we used a two-sided binomial test to determine whether a judge's choices differed significantly from a 50-50 split. The null hypothesis assumes that the judge is equally likely to choose either model's answer, meaning H<sub>0</sub>: p = 0.50. Ties were excluded from the test because the binomial test only compares two outcomes, choosing its own model or choosing the other model. Since the test is two-sided, it can detect a significant preference in either direction. However, we only considered the result evidence of self-preference when the p-value was below 0.05 and the judge selected its own model more often than the other model.

## 4 Results

### 4.1 Single Turn: Score level results

SP-Bias measures whether a model's own judge scores its answer differently from another judge when the answer itself is held fixed. Gemini showed the clearest score level self preference. The Gemini judge scored Gemini answers significantly higher than Claude, GPT, and Qwen did, with SP-Bias values of +0.29, +0.59, and +0.12, respectively. Qwen showed a similar pattern when compared with Claude and GPT, with significant differences of +0.22 and +0.63. However, the Qwen judge scored its answers lower than Gemini did, producing an SP-Bias of -0.14. Claude was generally more self-critical, scoring its own answers lower than Gemini and Qwen did. GPT was closer to being neutral. It scored its own answers significantly higher than Claude did (+0.23), but its differences relative to Gemini and Qwen were small and not statistically significant. Full score level results are reported in Table 2.

### 4.2 Single Turn: Decision level results

The own-pick rate measures how often a judge selected its own model's answer in a direct head-to-head comparison. Unlike SP-Bias, this comparison involves two different answers, so the result may reflect both self preference and genuine differences in answer quality. GPT showed the strongest self preference at the decision level, selecting its own answer 97% of the time against Claude, 96% against Gemini, and 99% against Qwen. All three comparisons were statistically significant. Claude generally preferred the competing model's answer, selecting itself only 31% of the time against GPT and 30% against Gemini. Its comparison with Qwen was nearly even, with Claude choosing itself 46% of the time. Qwen also rarely favored itself, particularly against GPT, where it selected its own answer only 26% of the time. Gemini showed a weaker self leaning pattern, selecting itself 57% of the time against Claude and 58% against Qwen, although only the comparison with Qwen was statistically significant. Against GPT, Gemini selected its own answer only 36% of the time. Full head-to-head results are shown in Table 3.

**Table 2. Single-turn score-level self-preference results.** One model answer is judged by two judges: its own model and an alternate model. A positive SP-Bias value indicates that a model's own judge scored its answer higher than the comparison judge did.

| Model Answer | Model Judge | Other Judge | SP-Bias | 95% CI | p | Significant? |
|---|---|---|---:|---|---:|:---:|
| claude-sonnet-5 | claude-sonnet-5 | gpt-5.5 | +0.17 | [+0.10, +0.23] | < 0.001 | Yes |
| claude-sonnet-5 | claude-sonnet-5 | gemini-3.1-flash-lite | -0.49 | [-0.57, -0.40] | < 0.001 | Yes |
| claude-sonnet-5 | claude-sonnet-5 | qwen3.6-35b | -0.12 | [-0.23, -0.02] | 0.022 | Yes |
| gpt-5.5 | gpt-5.5 | claude-sonnet-5 | +0.23 | [+0.18, +0.28] | < 0.001 | Yes |
| gpt-5.5 | gpt-5.5 | gemini-3.1-flash-lite | -0.02 | [-0.08, +0.04] | 0.555 | No |
| gpt-5.5 | gpt-5.5 | qwen3.6-35b | +0.07 | [-0.01, +0.14] | 0.084 | No |
| gemini-3.1-flash-lite | gemini-3.1-flash-lite | claude-sonnet-5 | +0.29 | [+0.22, +0.37] | < 0.001 | Yes |
| gemini-3.1-flash-lite | gemini-3.1-flash-lite | gpt-5.5 | +0.59 | [+0.53, +0.65] | < 0.001 | Yes |
| gemini-3.1-flash-lite | gemini-3.1-flash-lite | qwen3.6-35b | +0.12 | [+0.06, +0.18] | < 0.001 | Yes |
| qwen3.6-35b | qwen3.6-35b | claude-sonnet-5 | +0.22 | [+0.11, +0.32] | < 0.001 | Yes |
| qwen3.6-35b | qwen3.6-35b | gpt-5.5 | +0.63 | [+0.54, +0.72] | < 0.001 | Yes |
| qwen3.6-35b | qwen3.6-35b | gemini-3.1-flash-lite | -0.14 | [-0.23, -0.04] | 0.004 | Yes |

**Table 3. Single-turn decision-level self-preference results.** Two different model answers (own model and alternate model) are judged. OWN indicates the percentage of comparisons in which the judge selected its own model's answer. Significance is only shown when OWN is greater than 50% and the p-value is less than 0.05.

| Judge Model | Own Model Answer | Other Model Answer | n | OWN | OTHER | TIE | p(own = 0.5) | Significant? |
|---|---|---|---:|---:|---:|---:|---:|:---:|
| claude-sonnet-5 | claude-sonnet-5 | gpt-5.5 | 125 | 31% | 67% | 2% | < 0.001 | No |
| claude-sonnet-5 | claude-sonnet-5 | gemini-3.1-flash-lite | 125 | 30% | 66% | 3% | < 0.001 | No |
| claude-sonnet-5 | claude-sonnet-5 | qwen3.6-35b | 125 | 46% | 48% | 6% | 0.927 | No |
| gpt-5.5 | gpt-5.5 | claude-sonnet-5 | 125 | 97% | 2% | 1% | < 0.001 | Yes |
| gpt-5.5 | gpt-5.5 | gemini-3.1-flash-lite | 125 | 96% | 3% | 1% | < 0.001 | Yes |
| gpt-5.5 | gpt-5.5 | qwen3.6-35b | 125 | 99% | 1% | 0% | < 0.001 | Yes |
| gemini-3.1-flash-lite | gemini-3.1-flash-lite | claude-sonnet-5 | 125 | 57% | 40% | 3% | 0.069 | No |
| gemini-3.1-flash-lite | gemini-3.1-flash-lite | gpt-5.5 | 125 | 36% | 58% | 6% | 0.013 | No |
| gemini-3.1-flash-lite | gemini-3.1-flash-lite | qwen3.6-35b | 125 | 58% | 39% | 2% | 0.037 | Yes |
| qwen3.6-35b | qwen3.6-35b | claude-sonnet-5 | 125 | 49% | 49% | 2% | 1.000 | No |
| qwen3.6-35b | qwen3.6-35b | gpt-5.5 | 125 | 26% | 71% | 2% | < 0.001 | No |
| qwen3.6-35b | qwen3.6-35b | gemini-3.1-flash-lite | 125 | 42% | 54% | 4% | 0.235 | No |

### 4.3 Multi Turn: Score level results

In the multi-turn setting, SP-Bias measures whether a model's own judge scores its conversation differently from another judge when the same conversation is held fixed, calculated separately at 2, 4, 6, and 8 turns. Gemini showed the clearest score level self preference, scoring its own conversations higher than every other judge did at every conversation length. These differences were statistically significant across all comparisons. Qwen showed a similar pattern, consistently scoring its own conversations higher than Claude and GPT did. Its difference relative to Gemini was small at shorter lengths but increased to +0.29 by 8 turns. Claude was generally more self critical, scoring its own conversations lower than Gemini and Qwen did at every length. However, its comparison with GPT changed as conversations became longer, moving from -0.15 at 2 turns to +0.26 at 8 turns. GPT was close to neutral relative to Gemini and Qwen, but consistently scored its own conversations higher than Claude did. Overall, score level self preference tended to become more pronounced as conversation length increased, although this pattern was not identical for every model pair. Full results are shown in Table 4.

**Table 4. Multi-turn score-level self-preference results.** One model's conversation is judged by two judges, its own model and an alternate model, across conversation lengths of two, four, six, and eight turns. A positive SP-Bias value indicates that the model's own judge scored its conversation higher than the comparison judge did; a negative value indicates that its own judge scored it lower.

| Model Answer | Model Judge | Other Judge | 2 turns | 4 turns | 6 turns | 8 turns |
|---|---|---|---:|---:|---:|---:|
| Claude | Claude | GPT-5.5 | -0.15* | -0.10* | +0.10* | +0.26* |
| Claude | Claude | Gemini | -0.34* | -0.38* | -0.33* | -0.23* |
| Claude | Claude | Qwen | -0.24* | -0.34* | -0.23* | -0.16* |
| GPT-5.5 | GPT-5.5 | Claude | +0.35* | +0.41* | +0.43* | +0.46* |
| GPT-5.5 | GPT-5.5 | Gemini | -0.05* | +0.01 | +0.02 | +0.05* |
| GPT-5.5 | GPT-5.5 | Qwen | +0.02 | +0.03* | +0.04* | +0.00 |
| Gemini | Gemini | Claude | +0.44* | +0.59* | +0.44* | +0.43* |
| Gemini | Gemini | GPT-5.5 | +0.25* | +0.26* | +0.25* | +0.35* |
| Gemini | Gemini | Qwen | +0.18* | +0.18* | +0.16* | +0.12* |
| Qwen | Qwen | Claude | +0.42* | +0.47* | +0.55* | +0.54* |
| Qwen | Qwen | GPT-5.5 | +0.25* | +0.24* | +0.30* | +0.37* |
| Qwen | Qwen | Gemini | +0.01 | +0.03 | +0.10* | +0.29* |

*Note. \* p < 0.05.*

### 4.4 Multi Turn: Decision level results

The own-pick rate measures how often a judge selected its own model's conversation in a direct head-to-head comparison. GPT showed the strongest self preference at the decision level, choosing its own conversation between 82% and 100% of the time across all models and conversation lengths. These comparisons were statistically significant. Claude also strongly preferred its own conversations against Gemini and Qwen, selecting itself between 90% and 98% of the time, but it selected itself much less often against GPT, with own-pick rates ranging from 27% to 46%. Gemini and Qwen generally preferred the larger closed-source models' conversations. Gemini selected Claude or GPT in roughly 93% to 99% of comparisons and only showed a clear self-preference against Qwen, where its own-pick rate increased from 55% to 76% and became statistically significant at 6 and 8 turns. Qwen rarely selected itself against Claude or GPT and showed a more mixed pattern against Gemini. These differences generally became larger as conversations grew longer. However, because GPT and Claude were preferred by most judges overall, their high own-pick rates may reflect better outputs as well as self-preference. Full results are reported in Table 5.

**Table 5. Multi-turn decision-level self-preference results.** The table shows how often each judge selected its own model's conversation over an alternate model's conversation at two, four, six, and eight turns. Each percentage represents the judge's own-model selection rate at that conversation length. An asterisk indicates statistically significant self-preference, meaning the percentage is greater than 50% and the p-value is less than 0.05.

| Judge Model | Own Model Answer | Other Model Answer | 2 turns | 4 turns | 6 turns | 8 turns |
|---|---|---|---:|---:|---:|---:|
| Claude | Claude | GPT-5.5 | 38% | 27% | 33% | 46% |
| Claude | Claude | Gemini | 91%* | 93%* | 91%* | 93%* |
| Claude | Claude | Qwen | 90%* | 93%* | 95%* | 98%* |
| GPT-5.5 | GPT-5.5 | Claude | 82%* | 93%* | 96%* | 98%* |
| GPT-5.5 | GPT-5.5 | Gemini | 99%* | 100%* | 100%* | 100%* |
| GPT-5.5 | GPT-5.5 | Qwen | 100%* | 99%* | 100%* | 100%* |
| Gemini | Gemini | Claude | 5% | 2% | 1% | 2% |
| Gemini | Gemini | GPT-5.5 | 1% | 1% | 2% | 7% |
| Gemini | Gemini | Qwen | 55% | 54% | 61%* | 76%* |
| Qwen | Qwen | Claude | 8% | 3% | 5% | 2% |
| Qwen | Qwen | GPT-5.5 | 5% | 2% | 2% | 0% |
| Qwen | Qwen | Gemini | 52% | 58%* | 53% | 37% |

*Note. \* p < 0.05 indicates statistically significant self-preference.*

## 5 Discussion

As medical benchmarking and monitoring relies more on LLMs-as-a-judge, it has become increasingly important to understand how LLMs exhibit self-preference. Alvarez-Arenas et al. [2026] found that clinical LLM judges may favor AI-generated responses over human-written responses, partly because of response length and writing style. Panickssery et al. [2024] showed that LLM judges can recognize and favor their own generations on general tasks like news summarization, while Wataoka et al. [2025] found that judges tend to prefer lower-perplexity responses that feel more familiar to them. Our study focuses on healthcare, where benchmarking results can impact which models are selected for implementation. We examine self-preference across four different model families, separating scoring generosity from direct preferences for a model's own responses in head to head comparisons. Moreover, we look at whether multi-turn patient-physician conversations of increasing length, ranging from two to eight turns impact self preference.

### 5.1 Single Turn

In the single turn setting, we found that self-preference was evident and statistically significant for many comparisons. However, the strength of this preference varied across models. Gemini and Qwen inflated their own scores, as shown in Table 2, yet they often chose the rival model in head to head comparisons. GPT showed the opposite pattern. It displayed relatively little score inflation but selected its own answer in 96-99% of head to head comparisons. However, the GPT answers were also frequently preferred by the other judges, suggesting that its high selection rate may be due in part to stronger responses rather than self preference alone. Meanwhile, Claude showed little consistent self preference and often appeared more self critical than the other models.

These findings suggests that self preference can also be contingent on how the scoring is done, which is an additional important consideration. Thus, looking at only one measure can be misleading. Based only on decision level results, Gemini and Qwen may appear unbiased (no self preference), while score-level results alone may make GPT appear unbiased. The score level measure provides a better estimate of judge bias because both judges score the exact same answer from a single model, while the decision level measure is also affected by differences in answer quality. These findings suggest important considerations for those using LLMs-as-a-judge in medical evaluation. For example, pipelines should assess both a score level and decision level preference in order to assess congruence. Moreover, self preference could be mitigated by using a panel of judges from multiple model families rather than relying on a single judge or metric.

### 5.2 Multi-Turn

In the multi-turn setting, we found many of the same patterns that were observed in the single turn results. Gemini and Qwen generally inflated the scores of their own conversations, while GPT showed little score inflation but strongly preferred its own conversations in head to head comparisons. Again, Claude appeared more self critical in several comparisons. Since these patterns remained across a different dataset and evaluation format, they are less likely to be caused by the single turn setup alone. The score-level and decision-level results also continued to diverge, which shows that scoring generosity and direct preference are separate behaviors. GPT and Claude conversations were also frequently preferred by other judges, so their high self-pick rates may be because of stronger conversation quality rather than self-preference alone.

The main finding unique to the multi-turn setting was that self-preference generally became stronger as conversations grew from two to eight turns. In some comparisons, models showed little or no self-preference in shorter conversations but began favoring their own model as the conversations became longer. Longer conversations may give judges more opportunities to recognize and favor their own model family's questioning style or reasoning, particularly since all the comparisons here were done with the judge not being aware of the model. This also suggests that single turn benchmarks may underestimate the level of self preference that could appear in real conversational medical systems.

### 5.3 Limitations

There are limitations to our study. Here, we evaluated only one model from each model family. Different models within the same family, such as Claude Sonnet and Claude Opus, may show different self-preference patterns. The model self-preference was also not benchmarked against human evaluations which may lead to self preference being exhibited in some cases accurately; the preferred model consistently does have better responses. However, we believe that by evaluating model judging patterns at scale we can still find patterns of bias. This paper presents results for frontier model choices available at the time of the experiments but these results may not reflect evaluations on what may be the highest performing models in just a few months time. To address this, we plan to continuously evaluate self-preference of newer models upon release in the form of a public medical self preference leaderboard.

## 6 Conclusion

Overall, our findings show that self-preference is present in medical LLM judging, but it does not appear in the same way across all models. Gemini and Qwen mainly showed score-level self-preference, while GPT showed stronger decision-level self-preference. Claude was often more self-critical. We also found that how models are evaluated (scoring or pairwise selection) can lead to very different conclusions, which shows why evaluators must be careful in how they design their evaluation. In the multi-turn setting, self-preference generally became stronger as conversations grew longer, which suggests that single-turn benchmarks can underestimate the bias that can appear in real conversational medical systems. These results highlight the importance of using multiple judge families and multiple evaluation measures when comparing medical LLMs.

## References

Zheng, Lianmin, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." In *Advances in Neural Information Processing Systems*, volume 36, 2023.

Li, Lingyao, Deyi Li, Chen Chen, Renkai Ma, Runlong Yu, Mingquan Lin, Rui Yin, Lizhou Fan, Cathy Shyr, Siyuan Ma, Mei Liu, and Steven Bethard. "LLM-as-a-Judge in Healthcare: A Scoping Analysis of Applications, Methods, and Human Alignment." *arXiv preprint* arXiv:2605.25273, 2026. doi: 10.48550/arXiv.2605.25273.

Wataoka, Koki, Tsubasa Takahashi, and Ryokan Ri. "Self-Preference Bias in LLM-as-a-Judge." *arXiv preprint* arXiv:2410.21819, 2025. doi: 10.48550/arXiv.2410.21819.

Alvarez-Arenas, J. I., D. Jimenez-Carretero, D. Mananes, and F. Sanchez-Cabo. "The Unreliable Judges: Assessing Reproducibility and Self-Preference Bias of LLMs as Free-Text Evaluators." *medRxiv*, 2026. doi: 10.64898/2026.06.15.26355670. Preprint, version 2.

Pombal, Jose, Ricardo Rei, and Andre F. T. Martins. "Self-Preference Bias in Rubric-Based Evaluation of Large Language Models." *arXiv preprint* arXiv:2604.06996, 2026. doi: 10.48550/arXiv.2604.06996. Published as a conference paper at COLM 2026.

Panickssery, Arjun, Samuel R. Bowman, and Shi Feng. "LLM Evaluators Recognize and Favor Their Own Generations." In *Advances in Neural Information Processing Systems*, volume 37, 2024. doi: 10.52202/079017-2197.
