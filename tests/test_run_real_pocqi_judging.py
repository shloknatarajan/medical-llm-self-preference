"""Tests for the production Real-POCQi batch judging runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from generation.real_pocqi import GenerationStatus, RealPocqiOutput
from inference import ModelResponse, Provider, TokenUsage
from judging import (
    DirectRankingOutput,
    RealPocqiScores,
    ResponseRanking,
    RubricAndModelRankingOutput,
    RubricScoringOutput,
    ScoredPocqiResponse,
)
from judging.run_real_pocqi_judging import (
    load_latest_question_responses,
    parse_args,
    run,
)


def _generation(
    *,
    question: int,
    model: str,
    attempt: int = 1,
    response_suffix: str = "",
) -> RealPocqiOutput:
    return RealPocqiOutput(
        experiment_id="generation-experiment",
        run_id="generation-run",
        generation_key=f"question-{question}__{model}",
        generation_id=f"generation-{question}-{model}-{attempt}",
        attempt=attempt,
        status=GenerationStatus.SUCCEEDED,
        created_at=f"2026-08-24T12:00:0{attempt}+00:00",
        question_id=f"question-{question}",
        question_text=f"Clinical question {question}?",
        specialty="Medicine",
        generator_family="test-family",
        generator_model=model,
        prompt_template_id="generation-v1",
        system_prompt="system",
        user_prompt="user",
        response_text=f"Answer from {model}{response_suffix}",
    )


def _write_generations(
    path: Path,
    *,
    questions: int = 2,
    models: tuple[str, ...] = ("generator-a", "generator-b"),
) -> None:
    records = [
        _generation(question=question, model=model)
        for question in range(1, questions + 1)
        for model in models
    ]
    path.write_text(
        "\n".join(record.to_json() for record in records) + "\n",
        encoding="utf-8",
    )


class BatchFakeCaller:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, model: str, input: str, **kwargs) -> ModelResponse:
        self.calls += 1
        response_format = kwargs["response_format"]
        ranking = ResponseRanking(response_ids=["response-1", "response-2"])
        scored = [
            ScoredPocqiResponse(
                response_id=response_id,
                scores=RealPocqiScores(
                    accuracy=score,
                    clinical_utility=score,
                    source_quality=score,
                    verifiability=score,
                    completeness=score,
                ),
            )
            for response_id, score in (("response-1", 5), ("response-2", 4))
        ]
        if response_format is DirectRankingOutput:
            parsed = DirectRankingOutput(ranking=ranking)
        elif response_format is RubricScoringOutput:
            parsed = RubricScoringOutput(scored_responses=scored)
        else:
            parsed = RubricAndModelRankingOutput(
                scored_responses=scored,
                model_ranking=ranking,
            )
        return ModelResponse(
            text=parsed.model_dump_json(),
            provider=Provider.OPENAI,
            model=model,
            parsed=parsed,
            request_id=f"request-{self.calls}",
            finish_reason="completed",
            usage=TokenUsage(input_tokens=100, output_tokens=20),
        )


def test_loader_selects_latest_successful_attempt(tmp_path: Path) -> None:
    path = tmp_path / "generations.jsonl"
    records = [
        _generation(question=1, model="generator-a"),
        _generation(question=1, model="generator-a", attempt=2, response_suffix=" new"),
        _generation(question=1, model="generator-b"),
    ]
    path.write_text(
        "\n".join(record.to_json() for record in records) + "\n",
        encoding="utf-8",
    )

    loaded = load_latest_question_responses(
        path,
        generator_models=("generator-a", "generator-b"),
    )

    assert len(loaded) == 1
    assert loaded[0].responses[0].generation_id.endswith("-2")
    assert loaded[0].responses[0].response_text.endswith(" new")


def test_loader_rejects_incomplete_question_matrix(tmp_path: Path) -> None:
    path = tmp_path / "generations.jsonl"
    path.write_text(
        _generation(question=1, model="generator-a").to_json() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incomplete generation coverage"):
        load_latest_question_responses(
            path,
            generator_models=("generator-a", "generator-b"),
        )


def test_loader_selects_repeatable_seeded_question_sample(tmp_path: Path) -> None:
    path = tmp_path / "generations.jsonl"
    _write_generations(path, questions=12)

    first = load_latest_question_responses(
        path,
        generator_models=("generator-a", "generator-b"),
        num_questions=5,
        question_sample_seed=20260824,
    )
    repeated = load_latest_question_responses(
        path,
        generator_models=("generator-a", "generator-b"),
        num_questions=5,
        question_sample_seed=20260824,
    )
    different = load_latest_question_responses(
        path,
        generator_models=("generator-a", "generator-b"),
        num_questions=5,
        question_sample_seed=7,
    )

    first_ids = [question.question_id for question in first]
    assert first_ids == [question.question_id for question in repeated]
    assert first_ids != [question.question_id for question in different]
    assert first_ids != [f"question-{index}" for index in range(1, 6)]


def test_batch_runner_dry_run_and_resume(tmp_path: Path, capsys) -> None:
    generations = tmp_path / "generations.jsonl"
    output_dir = tmp_path / "judgements"
    _write_generations(generations)
    common_args = [
        "--input-generations",
        str(generations),
        "--output-dir",
        str(output_dir),
        "--generator-models",
        "generator-a",
        "generator-b",
        "--judge-models",
        "gpt-test",
        "claude-test",
        "--experiment-id",
        "judging-experiment",
        "--max-concurrency",
        "2",
        "--openai-concurrency",
        "1",
        "--anthropic-concurrency",
        "1",
        "--gemini-concurrency",
        "1",
        "--retry-delay-seconds",
        "0",
    ]

    dry_caller = BatchFakeCaller()
    assert run(parse_args([*common_args, "--dry-run"]), model_caller=dry_caller) == 0
    assert dry_caller.calls == 0
    assert "12 logical judgments, 12 pending, 0 skipped" in capsys.readouterr().out

    caller = BatchFakeCaller()
    first_args = parse_args([*common_args, "--run-id", "batch-run-1"])
    assert run(first_args, model_caller=caller) == 0
    assert caller.calls == 12
    manifest = json.loads(
        (output_dir / "batch-run-1.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["logical_judgments"] == {
        "total": 12,
        "pending_at_start": 12,
        "skipped_existing": 0,
        "succeeded": 12,
        "failed": 0,
    }

    resumed_caller = BatchFakeCaller()
    resumed_args = parse_args([*common_args, "--run-id", "batch-run-2"])
    assert run(resumed_args, model_caller=resumed_caller) == 0
    assert resumed_caller.calls == 0
    resumed_manifest = json.loads(
        (output_dir / "batch-run-2.manifest.json").read_text(encoding="utf-8")
    )
    assert resumed_manifest["logical_judgments"]["skipped_existing"] == 12
    assert resumed_manifest["logical_judgments"]["pending_at_start"] == 0


def test_batch_runner_can_run_only_combined_scoring_and_ranking(
    tmp_path: Path,
) -> None:
    generations = tmp_path / "generations.jsonl"
    output_dir = tmp_path / "judgements"
    _write_generations(generations, questions=1)
    caller = BatchFakeCaller()
    args = parse_args(
        [
            "--input-generations",
            str(generations),
            "--output-dir",
            str(output_dir),
            "--generator-models",
            "generator-a",
            "generator-b",
            "--judge-models",
            "gpt-test",
            "claude-test",
            "--judging-cases",
            "rubric_and_model_ranking",
            "--experiment-id",
            "combined-only",
            "--run-id",
            "combined-run",
            "--retry-delay-seconds",
            "0",
        ]
    )

    assert run(args, model_caller=caller) == 0
    assert caller.calls == 2
    assert not (output_dir / "direct_ranking.jsonl").exists()
    assert not (output_dir / "rubric_sum_ranking.jsonl").exists()
    assert len(
        (output_dir / "rubric_and_model_ranking.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 2
    manifest = json.loads(
        (output_dir / "combined-run.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["judging_cases"] == ["rubric_and_model_ranking"]
    assert manifest["logical_judgments"]["total"] == 2


def test_identity_revealed_runner_uses_seeded_sample_and_separate_output(
    tmp_path: Path,
) -> None:
    generations = tmp_path / "generations.jsonl"
    output_dir = tmp_path / "judgements"
    _write_generations(generations, questions=4)
    caller = BatchFakeCaller()
    args = parse_args(
        [
            "--input-generations",
            str(generations),
            "--output-dir",
            str(output_dir),
            "--generator-models",
            "generator-a",
            "generator-b",
            "--judge-models",
            "gpt-test",
            "--judging-cases",
            "rubric_and_model_ranking",
            "--num-questions",
            "2",
            "--question-sample-seed",
            "42",
            "--reveal-generator-identities",
            "--run-id",
            "revealed-run",
            "--retry-delay-seconds",
            "0",
        ]
    )

    assert run(args, model_caller=caller) == 0
    revealed_path = output_dir / "identity_revealed_rubric_and_model_ranking.jsonl"
    assert revealed_path.exists()
    assert not (output_dir / "rubric_and_model_ranking.jsonl").exists()
    records = [json.loads(line) for line in revealed_path.read_text().splitlines()]
    assert len(records) == 2
    assert all(record["identity_blinded"] is False for record in records)
    assert all("generator_model=" in record["user_prompt"] for record in records)

    manifest = json.loads(
        (output_dir / "revealed-run.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["question_count"] == 2
    assert manifest["question_sample_seed"] == 42
    assert manifest["identity_blinded"] is False
    assert manifest["experiment_id"] == "real_pocqi_identity_revealed_random200_v1"
    assert manifest["judgment_output_paths"]["rubric_and_model_ranking"] == str(
        revealed_path
    )


def test_batch_runner_supports_modal_judge_with_custom_caller(
    tmp_path: Path,
) -> None:
    generations = tmp_path / "generations.jsonl"
    output_dir = tmp_path / "judgements"
    _write_generations(generations, questions=1)
    caller = BatchFakeCaller()
    args = parse_args(
        [
            "--input-generations",
            str(generations),
            "--output-dir",
            str(output_dir),
            "--generator-models",
            "generator-a",
            "generator-b",
            "--judge-models",
            "modal/Qwen-test",
            "--judging-cases",
            "direct_ranking",
            "--experiment-id",
            "modal-direct",
            "--run-id",
            "modal-run",
            "--modal-concurrency",
            "1",
            "--retry-delay-seconds",
            "0",
        ]
    )

    assert run(args, model_caller=caller) == 0
    assert caller.calls == 1
    record = json.loads(
        (output_dir / "direct_ranking.jsonl").read_text(encoding="utf-8")
    )
    assert record["judge_family"] == "qwen"
    assert record["judge_model"] == "Qwen-test"
    manifest = json.loads(
        (output_dir / "modal-run.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["provider_concurrency"]["modal"] == 1
