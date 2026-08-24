"""Create model-level self-preference score artifacts from completed judgments.

The score definitions follow the original Med-Self-Preference draft. Rubric
SP-Bias holds an answer fixed and subtracts an outside judge's overall score
from the answer generator's own-judge score. Decision preference asks whether
the own judge ranks its answer above a competitor. Full-list rankings are
converted into all pairwise decisions, with no ties because the saved ranking
schema is strict.

Only the Python standard library is required.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data/analysis/self_preference"

CONDITIONS = (
    {
        "dataset": "real_pocqi",
        "condition": "rubric_and_ranking",
        "turns": None,
        "experiment_id": "real_pocqi_combined_all_judges_v1",
        "path": ROOT / "data/real_pcoqi/judgements/rubric_and_model_ranking.jsonl",
    },
    {
        "dataset": "real_pocqi",
        "condition": "direct_ranking",
        "turns": None,
        "experiment_id": "real_pocqi_direct_ranking_random100_v1",
        "path": ROOT / "data/real_pcoqi/judgements/direct_ranking.jsonl",
    },
    {
        "dataset": "real_pocqi",
        "condition": "identity_revealed_rubric_and_ranking",
        "turns": None,
        "experiment_id": "real_pocqi_identity_revealed_random200_v1",
        "path": ROOT
        / "data/real_pcoqi/judgements/identity_revealed_rubric_and_model_ranking.jsonl",
    },
    *(
        {
            "dataset": "medsp1000",
            "condition": "rubric_and_ranking",
            "turns": turns,
            "experiment_id": "medsp1000_all_judges_v1",
            "path": ROOT
            / "data/outputs/medsp1000/judgements"
            / (
                "rubric_and_model_ranking.jsonl"
                if turns == 8
                else f"rubric_and_model_ranking_{turns}_turns.jsonl"
            ),
        }
        for turns in (2, 4, 6, 8)
    ),
)


def read_records(path: Path, experiment_id: str) -> list[dict[str, Any]]:
    """Return the last successful attempt for each question-judge cell."""

    records: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("experiment_id") != experiment_id or row.get("status") != "succeeded":
                continue
            records[(row["question_id"], row["judge_model"])] = row
    return list(records.values())


def response_maps(row: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    response_to_model = {
        candidate["response_id"]: candidate["generator_model"]
        for candidate in row["candidates"]
    }
    model_to_response = {model: response for response, model in response_to_model.items()}
    if len(model_to_response) != len(response_to_model):
        raise ValueError(f"Duplicate generator model for question {row['question_id']}")
    return response_to_model, model_to_response


def values_for_row(row: dict[str, Any]) -> dict[str, dict[str, float | int | None]]:
    """Map each generator to its rank and mean rubric score in one judgment."""

    response_to_model, _ = response_maps(row)
    rank_block = row["result"].get("model_ranking") or row["result"].get("ranking")
    if not rank_block:
        raise ValueError(f"Missing ranking for question {row['question_id']}")
    ranks = {
        response_to_model[response_id]: rank
        for rank, response_id in enumerate(rank_block["response_ids"], 1)
    }
    overall_scores: dict[str, float] = {}
    for scored in row["result"].get("scored_responses", []):
        scores = scored["scores"]
        overall_scores[response_to_model[scored["response_id"]]] = fmean(scores.values())
    return {
        model: {"rank": rank, "overall_score": overall_scores.get(model)}
        for model, rank in ranks.items()
    }


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1e-300 if abs(d) < 1e-300 else d
    d = 1.0 / d
    h = d
    for iteration in range(1, 501):
        doubled = 2 * iteration
        coefficient = (
            iteration * (b - iteration) * x / ((qam + doubled) * (a + doubled))
        )
        d = 1.0 + coefficient * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + coefficient / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        h *= d * c
        coefficient = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + doubled) * (qap + doubled))
        )
        d = 1.0 + coefficient * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + coefficient / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_two_sided_p(t_statistic: float, degrees_freedom: int) -> float:
    x = degrees_freedom / (degrees_freedom + t_statistic * t_statistic)
    return regularized_beta(x, degrees_freedom / 2.0, 0.5)


def t_critical_975(degrees_freedom: int) -> float:
    low, high = 0.0, 20.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if student_t_two_sided_p(midpoint, degrees_freedom) > 0.05:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def paired_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {key: None for key in ("n", "mean", "sd", "se", "ci95_low", "ci95_high", "t", "p")}
    mean = fmean(values)
    if len(values) == 1:
        return {
            "n": 1,
            "mean": mean,
            "sd": None,
            "se": None,
            "ci95_low": None,
            "ci95_high": None,
            "t": None,
            "p": None,
        }
    sd = stdev(values)
    se = sd / math.sqrt(len(values))
    if se == 0.0:
        t_statistic = math.copysign(math.inf, mean) if mean else 0.0
        p_value = 0.0 if mean else 1.0
    else:
        t_statistic = mean / se
        p_value = student_t_two_sided_p(t_statistic, len(values) - 1)
    margin = t_critical_975(len(values) - 1) * se
    return {
        "n": len(values),
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "t": t_statistic,
        "p": p_value,
    }


def exact_binomial_two_sided_p(wins: int, trials: int) -> float | None:
    if trials == 0:
        return None
    tail = min(wins, trials - wins)
    # At p=0.5 the distribution is symmetric, so twice the smaller inclusive
    # tail equals scipy.stats.binomtest's two-sided result.
    probability = regularized_beta(0.5, trials - tail, tail + 1)
    return min(1.0, 2.0 * probability)


def exact_binomial_two_sided_log10_p(wins: int, trials: int) -> float | None:
    """Return log10(p), preserving results too small for a float."""

    if trials == 0:
        return None
    tail = min(wins, trials - wins)
    log_terms = [
        math.lgamma(trials + 1)
        - math.lgamma(k + 1)
        - math.lgamma(trials - k + 1)
        - trials * math.log(2.0)
        for k in range(tail + 1)
    ]
    largest = max(log_terms)
    log_tail = largest + math.log(math.fsum(math.exp(value - largest) for value in log_terms))
    log_p = min(0.0, math.log(2.0) + log_tail)
    return log_p / math.log(10.0)


def wilson_interval(wins: int, trials: int) -> tuple[float | None, float | None]:
    if trials == 0:
        return None, None
    z = 1.959963984540054
    proportion = wins / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2))
        / denominator
    )
    return center - margin, center + margin


def decision_summary(wins: int, trials: int) -> dict[str, float | int | None]:
    low, high = wilson_interval(wins, trials)
    return {
        "wins": wins,
        "losses": trials - wins,
        "ties": 0,
        "non_tied_n": trials,
        "own_pick_rate": wins / trials if trials else None,
        "ci95_low": low,
        "ci95_high": high,
        "binomial_p": exact_binomial_two_sided_p(wins, trials),
        "binomial_log10_p": exact_binomial_two_sided_log10_p(wins, trials),
    }


def condition_rows(
    condition: dict[str, Any], records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_question: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_question[record["question_id"]][record["judge_model"]] = record

    question_rows: list[dict[str, Any]] = []
    pair_observations: dict[tuple[str, str], dict[str, list[float] | int]] = defaultdict(
        lambda: {"sp_bias": [], "rank_effect": [], "wins": 0, "trials": 0}
    )
    model_question_values: dict[str, dict[str, list[float] | int]] = defaultdict(
        lambda: {
            "sp_bias": [],
            "rank_effect": [],
            "own_score": [],
            "outside_score": [],
            "own_rank": [],
            "outside_rank": [],
            "wins": 0,
            "trials": 0,
            "outside_judges": 0,
        }
    )

    for question_id, judge_records in sorted(by_question.items()):
        values = {judge: values_for_row(row) for judge, row in judge_records.items()}
        judge_models = sorted(values)
        generator_models = sorted(next(iter(values.values())))
        if any(
            set(judge_values) != set(generator_models) for judge_values in values.values()
        ):
            raise ValueError(f"Incomplete candidate set for question {question_id}")
        for model in judge_models:
            own = values[model][model]
            outside_judges = [judge for judge in judge_models if judge != model]
            competitors = [generator for generator in generator_models if generator != model]
            outside_ranks = [float(values[judge][model]["rank"]) for judge in outside_judges]
            own_rank = float(own["rank"])
            rank_effect = own_rank - fmean(outside_ranks)

            own_score = own["overall_score"]
            outside_scores = [values[judge][model]["overall_score"] for judge in outside_judges]
            has_scores = own_score is not None and all(score is not None for score in outside_scores)
            outside_score_mean = fmean(outside_scores) if has_scores else None
            sp_bias = float(own_score) - outside_score_mean if has_scores else None

            own_wins = sum(
                own_rank < float(values[model][opponent]["rank"])
                for opponent in competitors
            )
            base = {
                "dataset": condition["dataset"],
                "condition": condition["condition"],
                "turns": condition["turns"],
                "experiment_id": condition["experiment_id"],
                "question_id": question_id,
                "model": model,
            }
            question_rows.append(
                {
                    **base,
                    "outside_judges": len(outside_judges),
                    "own_overall_score": own_score,
                    "outside_overall_score_mean": outside_score_mean,
                    "rubric_sp_bias": sp_bias,
                    "own_rank": own_rank,
                    "outside_rank_mean": fmean(outside_ranks),
                    "rank_self_preference_effect": rank_effect,
                    "decision_own_picks": own_wins,
                    "decision_comparisons": len(competitors),
                    "decision_own_pick_rate": own_wins / len(competitors),
                }
            )

            aggregate = model_question_values[model]
            if sp_bias is not None:
                aggregate["sp_bias"].append(sp_bias)  # type: ignore[union-attr]
                aggregate["own_score"].append(float(own_score))  # type: ignore[union-attr]
                aggregate["outside_score"].append(outside_score_mean)  # type: ignore[union-attr]
            aggregate["rank_effect"].append(rank_effect)  # type: ignore[union-attr]
            aggregate["own_rank"].append(own_rank)  # type: ignore[union-attr]
            aggregate["outside_rank"].append(fmean(outside_ranks))  # type: ignore[union-attr]
            aggregate["wins"] += own_wins  # type: ignore[operator]
            aggregate["trials"] += len(competitors)  # type: ignore[operator]
            aggregate["outside_judges"] += len(outside_judges)  # type: ignore[operator]

            for outside_judge in outside_judges:
                pair = pair_observations[(model, outside_judge)]
                pair["rank_effect"].append(
                    own_rank - float(values[outside_judge][model]["rank"])
                )  # type: ignore[union-attr]
                if has_scores:
                    pair["sp_bias"].append(
                        float(own_score) - float(values[outside_judge][model]["overall_score"])
                    )  # type: ignore[union-attr]
            for opponent in competitors:
                pair = pair_observations[(model, opponent)]
                pair["wins"] += int(
                    own_rank < float(values[model][opponent]["rank"])
                )  # type: ignore[operator]
                pair["trials"] += 1  # type: ignore[operator]

    common = {
        "dataset": condition["dataset"],
        "condition": condition["condition"],
        "turns": condition["turns"],
        "experiment_id": condition["experiment_id"],
    }
    model_rows = []
    for model, aggregate in sorted(model_question_values.items()):
        sp = paired_summary(aggregate["sp_bias"])  # type: ignore[arg-type]
        rank = paired_summary(aggregate["rank_effect"])  # type: ignore[arg-type]
        decision = decision_summary(int(aggregate["wins"]), int(aggregate["trials"]))
        model_rows.append(
            {
                **common,
                "model": model,
                "n_questions": len(aggregate["rank_effect"]),  # type: ignore[arg-type]
                "n_outside_models": int(aggregate["outside_judges"])
                // len(aggregate["rank_effect"]),  # type: ignore[arg-type]
                "own_overall_score_mean": mean_or_none(aggregate["own_score"]),  # type: ignore[arg-type]
                "outside_overall_score_mean": mean_or_none(aggregate["outside_score"]),  # type: ignore[arg-type]
                **prefixed("rubric_sp_bias", sp),
                "own_rank_mean": mean_or_none(aggregate["own_rank"]),  # type: ignore[arg-type]
                "outside_rank_mean": mean_or_none(aggregate["outside_rank"]),  # type: ignore[arg-type]
                **prefixed("rank_effect", rank),
                **prefixed("decision", decision),
            }
        )

    pair_rows = []
    for (model, outside_model), aggregate in sorted(pair_observations.items()):
        pair_rows.append(
            {
                **common,
                "model": model,
                "outside_model": outside_model,
                **prefixed("rubric_sp_bias", paired_summary(aggregate["sp_bias"])),  # type: ignore[arg-type]
                **prefixed("rank_effect", paired_summary(aggregate["rank_effect"])),  # type: ignore[arg-type]
                **prefixed(
                    "decision",
                    decision_summary(int(aggregate["wins"]), int(aggregate["trials"])),
                ),
            }
        )
    return model_rows, pair_rows, question_rows


def mean_or_none(values: list[float]) -> float | None:
    return fmean(values) if values else None


def prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty analysis file {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    sources = []
    for condition in CONDITIONS:
        records = read_records(condition["path"], condition["experiment_id"])
        if not records:
            raise ValueError(f"No completed records for {condition['experiment_id']}")
        models, pairs, questions = condition_rows(condition, records)
        model_rows.extend(models)
        pair_rows.extend(pairs)
        question_rows.extend(questions)
        sources.append(
            {
                "dataset": condition["dataset"],
                "condition": condition["condition"],
                "turns": condition["turns"],
                "experiment_id": condition["experiment_id"],
                "path": str(condition["path"].relative_to(ROOT)),
                "sha256": sha256(condition["path"]),
                "successful_logical_judgments": len(records),
            }
        )

    outputs = {
        "model_scores.csv": model_rows,
        "pairwise_scores.csv": pair_rows,
        "question_model_scores.csv": question_rows,
    }
    for filename, rows in outputs.items():
        write_csv(OUTPUT_DIR / filename, rows)

    manifest = {
        "methodology": "docs/archive/legacy_manuscript.md sections 3.1-3.3",
        "generator_script": str(Path(__file__).relative_to(ROOT)),
        "sources": sources,
        "outputs": {
            filename: {"rows": len(rows), "sha256": sha256(OUTPUT_DIR / filename)}
            for filename, rows in outputs.items()
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
