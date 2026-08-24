"""Analyze whether candidate length confounds model scoring or self-preference.

The primary length measure is a provider-neutral lexical token count computed
from the exact text shown to judges. Provider-reported output-token counts are
retained only as a sensitivity descriptor because providers use different
tokenizers.

For model scoring, fixed-effects regressions absorb the question-judge list,
generator model, and presentation position. The reported coefficient is the
change in rank or within-list standardized rubric score per doubling of answer
length. For self-preference, the exact answer is already held fixed by the
matched estimand. A separate moderation regression asks whether a model's
longer-than-usual answers have a different own-minus-outside judge effect.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data/analysis/self_preference"
REAL_GENERATIONS = ROOT / "data/outputs/generations/real_pocqi_generations.jsonl"
MED_GENERATIONS = ROOT / "data/outputs/medsp1000/generations.jsonl"
REAL_JUDGMENTS = ROOT / "data/real_pcoqi/judgements/rubric_and_model_ranking.jsonl"
MED_JUDGMENTS = {
    2: ROOT / "data/outputs/medsp1000/judgements/rubric_and_model_ranking_2_turns.jsonl",
    4: ROOT / "data/outputs/medsp1000/judgements/rubric_and_model_ranking_4_turns.jsonl",
    6: ROOT / "data/outputs/medsp1000/judgements/rubric_and_model_ranking_6_turns.jsonl",
    8: ROOT / "data/outputs/medsp1000/judgements/rubric_and_model_ranking.jsonl",
}
TOKEN_PATTERN = re.compile(r"\w+(?:['’\-]\w+)*|[^\w\s]", re.UNICODE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_judgments(path: Path, experiment_id: str) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("experiment_id") == experiment_id and row.get("status") == "succeeded":
            latest[(row["question_id"], row["judge_model"])] = row
    return list(latest.values())


def generation_index(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["generation_id"]: row
        for row in read_jsonl(path)
        if row.get("status") == "succeeded"
    }


def lexical_tokens(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def candidate_lengths(
    records: list[dict[str, Any]],
    generations: dict[str, dict[str, Any]],
    visible_turns: int | None,
) -> dict[tuple[str, str], dict[str, float]]:
    lengths: dict[tuple[str, str], dict[str, float]] = {}
    for row in records:
        for candidate in row["candidates"]:
            key = (row["question_id"], candidate["generator_model"])
            generation = generations[candidate["generation_id"]]
            if visible_turns is None:
                text = generation["response_text"]
                provider_tokens = generation.get("output_tokens")
                all_text = text
            else:
                prefix = generation["turns"][:visible_turns]
                clinician_turns = [turn for turn in prefix if turn["role"] == "clinician"]
                text = "\n".join(turn["content"] for turn in clinician_turns)
                all_text = "\n".join(turn["content"] for turn in prefix)
                provider_tokens = sum((turn.get("output_tokens") or 0) for turn in clinician_turns)
            value = {
                "lexical_tokens": float(lexical_tokens(text)),
                "trajectory_lexical_tokens": float(lexical_tokens(all_text)),
                "provider_tokens": float(provider_tokens) if provider_tokens is not None else math.nan,
            }
            previous = lengths.setdefault(key, value)
            if previous != value:
                raise ValueError(f"Inconsistent length for {key}")
    return lengths


def candidate_maps(row: dict[str, Any]) -> tuple[dict[str, str], dict[str, int]]:
    generator = {c["response_id"]: c["generator_model"] for c in row["candidates"]}
    position = {c["response_id"]: i + 1 for i, c in enumerate(row["candidates"])}
    return generator, position


def long_outcomes(
    records: list[dict[str, Any]], lengths: dict[tuple[str, str], dict[str, float]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranks: list[dict[str, Any]] = []
    rubrics: list[dict[str, Any]] = []
    for row in records:
        generator, position = candidate_maps(row)
        ranking = (row["result"].get("model_ranking") or row["result"]["ranking"])[
            "response_ids"
        ]
        list_id = f'{row["question_id"]}|{row["judge_model"]}'
        for rank, response_id in enumerate(ranking, 1):
            model = generator[response_id]
            length = lengths[(row["question_id"], model)]
            ranks.append(
                {
                    "question": row["question_id"],
                    "list": list_id,
                    "judge": row["judge_model"],
                    "generator": model,
                    "position": str(position[response_id]),
                    "log2_tokens": math.log2(length["lexical_tokens"]),
                    "log2_provider_tokens": math.log2(length["provider_tokens"]),
                    "outcome": float(rank),
                }
            )
        scores = {
            item["response_id"]: fmean(item["scores"].values())
            for item in row["result"]["scored_responses"]
        }
        mean = fmean(scores.values())
        variance = fmean((score - mean) ** 2 for score in scores.values())
        scale = math.sqrt(variance)
        for response_id, score in scores.items():
            model = generator[response_id]
            length = lengths[(row["question_id"], model)]
            rubrics.append(
                {
                    "question": row["question_id"],
                    "list": list_id,
                    "judge": row["judge_model"],
                    "generator": model,
                    "position": str(position[response_id]),
                    "log2_tokens": math.log2(length["lexical_tokens"]),
                    "log2_provider_tokens": math.log2(length["provider_tokens"]),
                    "outcome": 0.0 if scale == 0 else (score - mean) / scale,
                }
            )
    return ranks, rubrics


def position_adjust(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[(row["judge"], row["generator"])].append(row)
    positions = sorted({row["position"] for row in rows})
    position_effect = {position: 0.0 for position in positions}
    for _ in range(10_000):
        cell_effect = {
            cell: fmean(item["outcome"] - position_effect[item["position"]] for item in members)
            for cell, members in cells.items()
        }
        by_position: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            by_position[row["position"]].append(
                row["outcome"] - cell_effect[(row["judge"], row["generator"])]
            )
        updated = {position: fmean(by_position[position]) for position in positions}
        center = fmean(updated[row["position"]] for row in rows)
        updated = {position: value - center for position, value in updated.items()}
        change = max(abs(updated[p] - position_effect[p]) for p in positions)
        position_effect = updated
        if change < 1e-12:
            break
    return [
        {**row, "outcome": row["outcome"] - position_effect[row["position"]]}
        for row in rows
    ]


def matched_effect_rows(
    rows: list[dict[str, Any]], lengths: dict[tuple[str, str], dict[str, float]], rank: bool
) -> list[dict[str, Any]]:
    by_answer: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_answer[(row["question"], row["generator"])][row["judge"]] = row["outcome"]
    effects = []
    for (question, generator), judge_values in by_answer.items():
        if generator not in judge_values or len(judge_values) < 2:
            continue
        outside = [value for judge, value in judge_values.items() if judge != generator]
        own_minus_outside = judge_values[generator] - fmean(outside)
        effects.append(
            {
                "question": question,
                "model": generator,
                "log2_tokens": math.log2(lengths[(question, generator)]["lexical_tokens"]),
                "log2_provider_tokens": math.log2(
                    lengths[(question, generator)]["provider_tokens"]
                ),
                "effect": own_minus_outside if rank else -own_minus_outside,
            }
        )
    return effects


def residualize(
    values: list[float], rows: list[dict[str, Any]], factors: tuple[str, ...]
) -> list[float]:
    overall_mean = fmean(values)
    residuals = [value - overall_mean for value in values]
    factor_groups: list[dict[str, list[int]]] = []
    for factor in factors:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            groups[str(row[factor])].append(index)
        factor_groups.append(groups)
    for _ in range(1_000):
        largest = 0.0
        for groups in factor_groups:
            means = {group: fmean(residuals[i] for i in indexes) for group, indexes in groups.items()}
            largest = max(largest, max(abs(value) for value in means.values()))
            for group, indexes in groups.items():
                mean = means[group]
                for index in indexes:
                    residuals[index] -= mean
        if largest < 1e-12:
            break
    return residuals


def clustered_slope(
    rows: list[dict[str, Any]],
    outcome: str,
    factors: tuple[str, ...],
    predictor: str = "log2_tokens",
) -> dict[str, float | int | list[float]]:
    x = residualize([row[predictor] for row in rows], rows, factors)
    y = residualize([row[outcome] for row in rows], rows, factors)
    denominator = sum(value * value for value in x)
    beta = sum(a * b for a, b in zip(x, y, strict=True)) / denominator
    errors = [b - beta * a for a, b in zip(x, y, strict=True)]
    cluster_scores: dict[str, float] = defaultdict(float)
    for row, x_value, error in zip(rows, x, errors, strict=True):
        cluster_scores[row["question"]] += x_value * error
    clusters = len(cluster_scores)
    parameters = 1 + sum(len({str(row[factor]) for row in rows}) - 1 for factor in factors)
    correction = (clusters / (clusters - 1)) * ((len(rows) - 1) / (len(rows) - parameters))
    variance = correction * sum(score * score for score in cluster_scores.values()) / denominator**2
    se = math.sqrt(variance)
    p = math.erfc(abs(beta / se) / math.sqrt(2.0))
    return {
        "n_observations": len(rows),
        "n_question_clusters": clusters,
        "slope_per_length_doubling": beta,
        "clustered_se": se,
        "ci95": [beta - 1.96 * se, beta + 1.96 * se],
        "p_normal": p,
    }


def correlation(x: list[float], y: list[float]) -> float:
    x_mean, y_mean = fmean(x), fmean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y)
    )
    return numerator / denominator


def summarize_condition(
    dataset: str,
    turns: int | None,
    records: list[dict[str, Any]],
    lengths: dict[tuple[str, str], dict[str, float]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranks, rubrics = long_outcomes(records, lengths)
    scoring_rank = clustered_slope(ranks, "outcome", ("list", "generator", "position"))
    scoring_rubric = clustered_slope(rubrics, "outcome", ("list", "generator", "position"))
    rank_effects = matched_effect_rows(position_adjust(ranks), lengths, rank=True)
    rubric_effects = matched_effect_rows(position_adjust(rubrics), lengths, rank=False)
    rank_moderation = clustered_slope(rank_effects, "effect", ("model",))
    rubric_moderation = clustered_slope(rubric_effects, "effect", ("model",))
    provider_sensitivity = {
        "model_scoring_association": {
            "rank": clustered_slope(
                ranks, "outcome", ("list", "generator", "position"), "log2_provider_tokens"
            ),
            "rubric_z": clustered_slope(
                rubrics, "outcome", ("list", "generator", "position"), "log2_provider_tokens"
            ),
        },
        "self_preference_length_moderation": {
            "rank": clustered_slope(
                rank_effects, "effect", ("model",), "log2_provider_tokens"
            ),
            "rubric_z": clustered_slope(
                rubric_effects, "effect", ("model",), "log2_provider_tokens"
            ),
        },
    }

    by_model: dict[str, list[tuple[str, dict[str, float]]]] = defaultdict(list)
    for (question, model), value in lengths.items():
        by_model[model].append((question, value))
    mean_log_tokens = fmean(row["log2_tokens"] for row in ranks)
    rank_beta = float(scoring_rank["slope_per_length_doubling"])
    rubric_beta = float(scoring_rubric["slope_per_length_doubling"])
    model_rows = []
    for model in sorted(by_model):
        unique_lengths = [value for _, value in by_model[model]]
        model_ranks = [row for row in ranks if row["generator"] == model]
        model_rubrics = [row for row in rubrics if row["generator"] == model]
        model_rows.append(
            {
                "dataset": dataset,
                "turns": turns,
                "model": model,
                "n_questions": len(unique_lengths),
                "mean_lexical_tokens": fmean(v["lexical_tokens"] for v in unique_lengths),
                "median_lexical_tokens": median(v["lexical_tokens"] for v in unique_lengths),
                "mean_provider_tokens": fmean(v["provider_tokens"] for v in unique_lengths),
                "mean_rank": fmean(row["outcome"] for row in model_ranks),
                "length_adjusted_mean_rank": fmean(
                    row["outcome"] - rank_beta * (row["log2_tokens"] - mean_log_tokens)
                    for row in model_ranks
                ),
                "mean_rubric_z": fmean(row["outcome"] for row in model_rubrics),
                "length_adjusted_mean_rubric_z": fmean(
                    row["outcome"] - rubric_beta * (row["log2_tokens"] - mean_log_tokens)
                    for row in model_rubrics
                ),
            }
        )
    unique_values = [value for members in by_model.values() for _, value in members]
    return (
        {
            "dataset": dataset,
            "turns": turns,
            "questions": len({question for question, _ in lengths}),
            "models": len(by_model),
            "length_measure": "provider-neutral lexical tokens in clinician/answer text",
            "provider_vs_lexical_log_correlation": correlation(
                [math.log2(v["provider_tokens"]) for v in unique_values],
                [math.log2(v["lexical_tokens"]) for v in unique_values],
            ),
            "model_scoring_association": {
                "rank": scoring_rank,
                "rubric_z": scoring_rubric,
            },
            "self_preference_length_moderation": {
                "rank": rank_moderation,
                "rubric_z": rubric_moderation,
            },
            "provider_reported_token_sensitivity": provider_sensitivity,
        },
        model_rows,
    )


def main() -> None:
    real_generations = generation_index(REAL_GENERATIONS)
    med_generations = generation_index(MED_GENERATIONS)
    conditions: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []

    real_records = read_judgments(REAL_JUDGMENTS, "real_pocqi_combined_all_judges_v1")
    real_lengths = candidate_lengths(real_records, real_generations, None)
    summary, rows = summarize_condition("real_pocqi", None, real_records, real_lengths)
    conditions.append(summary)
    model_rows.extend(rows)

    for turns, path in MED_JUDGMENTS.items():
        records = read_judgments(path, "medsp1000_all_judges_v1")
        lengths = candidate_lengths(records, med_generations, turns)
        summary, rows = summarize_condition("medsp1000", turns, records, lengths)
        conditions.append(summary)
        model_rows.extend(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "method": {
            "primary_length": (
                "Provider-neutral lexical tokens computed from exact answer text; for MedSP1000, "
                "clinician turns only within each judged prefix."
            ),
            "model_scoring": (
                "OLS slope per length doubling after absorbing question-judge list, generator-model, "
                "and presentation-position fixed effects; standard errors clustered by question."
            ),
            "self_preference": (
                "Within-model moderation of the matched own-minus-outside effect; the exact answer, "
                "and therefore its length, is already held fixed in the primary contrast."
            ),
            "caution": (
                "Associations are observational and may reflect answer quality or style; provider token "
                "counts are not used as the primary cross-provider measure because tokenizers differ."
            ),
        },
        "conditions": conditions,
    }
    json_path = OUTPUT_DIR / "token_length_analysis.json"
    json_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = OUTPUT_DIR / "token_length_model_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(model_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(model_rows)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
