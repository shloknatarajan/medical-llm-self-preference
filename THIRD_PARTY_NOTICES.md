# Third-party data and attribution

The MIT License in this repository applies to repository-authored code and
documentation. It does not replace the terms attached to upstream datasets or
model-provider outputs.

## Real-POCQi

- Source: `jjfenglab/Real-POCQi`
- Frozen revision: `9002e1ddff506d354f1b7becc1213b96299d07f6`
- Upstream license: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
- Dataset: [huggingface.co/datasets/jjfenglab/Real-POCQi](https://huggingface.co/datasets/jjfenglab/Real-POCQi)
- Paper: Jean Feng, Vishal Patel, Patrick Heagerty, Yifan Mai, Venkatesh
  Sivaraman, Patrick Vossler, Jialin Ouyang, and Anupam B. Jena, “Expert
  Evaluation of Clinical AI Tools on Real Point-of-Care Clinical Queries,”
  [arXiv:2606.28960](https://arxiv.org/abs/2606.28960), 2026.

This repository converts the upstream 620-row `questions.parquet` split into a
deterministic JSONL question set with source revision, row index, specialty,
and content provenance. It does not redistribute the upstream answer or
physician-rating splits. New model generations and judgments are stored
separately from the attributed questions.

## MedSP1000

- Source: `byrLLCC/MedSP1000`
- Frozen revision: `55e3e55efd08c73baab912ba0c5b42637114fbc8`
- Upstream dataset license: [MIT](https://opensource.org/license/mit)
- Dataset: [huggingface.co/datasets/byrLLCC/MedSP1000](https://huggingface.co/datasets/byrLLCC/MedSP1000)
- Paper: Cheng Liang, Pengcheng Qiu, Ya Zhang, Yanfeng Wang, Chaoyi Wu, and
  Weidi Xie, “Evaluating Large Language Models in Dynamic Clinical
  Decision-Making with Standardized Patient Cases,”
  [arXiv:2606.05112](https://arxiv.org/abs/2606.05112), 2026.

MedSP1000 is derived from peer-reviewed MedEdPORTAL teaching materials. This
repository deterministically selects 200 scenarios, preserves separate
clinician-visible and standardized-patient contexts, and excludes evaluator
and environment-controller materials from model inputs. The frozen JSONL
records retain the upstream dataset identifier and revision.

## Model outputs

Generated answers, simulated conversations, and LLM judgments are retained for
scientific auditability. They may contain factual errors or unsafe medical
content and must not be used for patient care. Their use may also be governed
by the terms of the model providers and model-weight licensors identified in
the saved run manifests.
