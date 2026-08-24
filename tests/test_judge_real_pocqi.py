"""End-to-end tests for the three-condition Real-POCQi judge."""

from __future__ import annotations

from pathlib import Path

from inference import ModelResponse, Provider, TokenUsage
from judging import (
    DirectRankingOutput,
    PocqiJudgingCase,
    PocqiJudgingSettings,
    PocqiJudgmentRecord,
    PocqiJudgmentStatus,
    PocqiResponseInput,
    RealPocqiScores,
    ResponseRanking,
    RubricAndModelRankingOutput,
    RubricScoringOutput,
    ScoredPocqiResponse,
    judge_pocqi_responses,
    load_pocqi_resume_state,
)


class FakeJudgeCaller:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, model: str, input: str, **kwargs) -> ModelResponse:
        self.calls.append({"model": model, "input": input, **kwargs})
        response_format = kwargs["response_format"]
        ranking = ResponseRanking(
            response_ids=["response-2", "response-1", "response-3"]
        )
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
            for response_id, score in (
                ("response-1", 4),
                ("response-2", 5),
                ("response-3", 3),
            )
        ]
        if response_format is DirectRankingOutput:
            parsed = DirectRankingOutput(ranking=ranking)
        elif response_format is RubricScoringOutput:
            parsed = RubricScoringOutput(scored_responses=scored)
        else:
            parsed = RubricAndModelRankingOutput(
                scored_responses=scored,
                model_ranking=ResponseRanking(
                    response_ids=["response-1", "response-2", "response-3"]
                ),
            )
        return ModelResponse(
            text=parsed.model_dump_json(),
            parsed=parsed,
            provider=Provider.OPENAI,
            model="gpt-test-version",
            request_id=f"request-{len(self.calls)}",
            finish_reason="completed",
            usage=TokenUsage(input_tokens=100, output_tokens=20),
        )


def _responses() -> list[PocqiResponseInput]:
    return [
        PocqiResponseInput(
            generation_id=f"generation-{index}",
            generator_family=family,
            generator_model=model,
            response_text=f"Clinical answer {index}",
        )
        for index, (family, model) in enumerate(
            (
                ("openai", "gpt-test"),
                ("anthropic", "claude-test"),
                ("google", "gemini-test"),
            ),
            start=1,
        )
    ]


def test_three_case_judging_smoke(tmp_path: Path) -> None:
    caller = FakeJudgeCaller()
    output_paths = {
        case: tmp_path / f"{case.value}.jsonl" for case in PocqiJudgingCase
    }

    records = judge_pocqi_responses(
        question_id="question-1",
        question_text="What is the recommended treatment?",
        specialty="Cardiology",
        responses=_responses(),
        judge_model="openai/gpt-test",
        settings=PocqiJudgingSettings(
            experiment_id="experiment-1",
            run_id="run-1",
            presentation_seed=42,
        ),
        model_caller=caller,
        output_paths=output_paths,
    )

    assert list(records) == list(PocqiJudgingCase)
    assert len(caller.calls) == 3
    assert caller.calls[0]["response_format"] is DirectRankingOutput
    assert caller.calls[1]["response_format"] is RubricScoringOutput
    assert caller.calls[2]["response_format"] is RubricAndModelRankingOutput
    assert "accuracy" not in caller.calls[0]["system"]
    assert "Without using a predefined rubric" in caller.calls[0]["input"]
    assert "accuracy" in caller.calls[1]["system"]
    assert "Do not provide a ranking" in caller.calls[1]["input"]
    for call in caller.calls:
        prompt_seen_by_judge = call["system"] + "\n" + call["input"]
        for hidden_value in (
            "generation-1",
            "generation-2",
            "generation-3",
            "gpt-test",
            "claude-test",
            "gemini-test",
            "openai",
            "anthropic",
            "google",
        ):
            assert hidden_value not in prompt_seen_by_judge

    presented_orders = [
        [candidate.generation_id for candidate in record.candidates]
        for record in records.values()
    ]
    assert presented_orders[0] == presented_orders[1] == presented_orders[2]

    for case, path in output_paths.items():
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        restored = PocqiJudgmentRecord.model_validate_json(lines[0])
        assert restored.status is PocqiJudgmentStatus.SUCCEEDED
        assert restored.judging_case is case

    rubric_record = records[PocqiJudgingCase.RUBRIC_SUM_RANKING]
    assert rubric_record.result.ranking.response_ids == [
        "response-2",
        "response-1",
        "response-3",
    ]
    combined_record = records[PocqiJudgingCase.RUBRIC_AND_MODEL_RANKING]
    assert combined_record.result.model_ranking.response_ids == [
        "response-1",
        "response-2",
        "response-3",
    ]
    assert combined_record.result.score_sum_ranking.response_ids == [
        "response-2",
        "response-1",
        "response-3",
    ]

    model_by_response_id = {
        candidate.response_id: candidate.generator_model
        for candidate in combined_record.candidates
    }
    assert combined_record.resolved_rankings() == {
        "model_ranking": [
            model_by_response_id["response-1"],
            model_by_response_id["response-2"],
            model_by_response_id["response-3"],
        ],
        "score_sum_ranking": [
            model_by_response_id["response-2"],
            model_by_response_id["response-1"],
            model_by_response_id["response-3"],
        ],
    }

    # The same logical run resumes without another model call or JSONL append.
    resumed = judge_pocqi_responses(
        question_id="question-1",
        question_text="What is the recommended treatment?",
        specialty="Cardiology",
        responses=list(reversed(_responses())),
        judge_model="openai/gpt-test",
        settings=PocqiJudgingSettings(
            experiment_id="experiment-1",
            run_id="a-new-run-id",
            presentation_seed=42,
        ),
        model_caller=caller,
        output_paths=output_paths,
    )

    assert len(caller.calls) == 3
    assert {
        case: record.judgment_id for case, record in resumed.items()
    } == {
        case: record.judgment_id for case, record in records.items()
    }
    assert all(
        len(path.read_text(encoding="utf-8").splitlines()) == 1
        for path in output_paths.values()
    )
    for case, path in output_paths.items():
        state = load_pocqi_resume_state(path)
        key = records[case].resolved_judgment_key()
        assert state.succeeded_by_key[key] == records[case]
        assert state.highest_attempt_by_key[key] == 1


def test_identity_revealed_judging_shows_models_and_records_condition(
    tmp_path: Path,
) -> None:
    caller = FakeJudgeCaller()
    output_path = tmp_path / "identity_revealed_rubric_and_model_ranking.jsonl"

    records = judge_pocqi_responses(
        question_id="question-1",
        question_text="What is the recommended treatment?",
        specialty="Cardiology",
        responses=_responses(),
        judge_model="openai/gpt-test",
        settings=PocqiJudgingSettings(
            experiment_id="identity-revealed-test",
            run_id="run-1",
        ),
        model_caller=caller,
        output_paths={
            PocqiJudgingCase.RUBRIC_AND_MODEL_RANKING: output_path,
        },
        judging_cases=(PocqiJudgingCase.RUBRIC_AND_MODEL_RANKING,),
        reveal_generator_identities=True,
    )

    assert len(caller.calls) == 1
    visible = caller.calls[0]["system"] + "\n" + caller.calls[0]["input"]
    for model in ("gpt-test", "claude-test", "gemini-test"):
        assert f'generator_model="{model}"' in visible
    assert "explicitly supplied for each candidate" in caller.calls[0]["system"]
    assert "Do not infer or identify" not in caller.calls[0]["system"]
    for hidden in ("generation-1", "generation-2", "generation-3"):
        assert hidden not in visible

    record = records[PocqiJudgingCase.RUBRIC_AND_MODEL_RANKING]
    assert record.identity_blinded is False
    assert "identity_revealed" in record.prompt_template_id
    restored = PocqiJudgmentRecord.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert restored.identity_blinded is False


def test_force_appends_a_new_attempt_for_every_case(tmp_path: Path) -> None:
    caller = FakeJudgeCaller()
    output_paths = {
        case: tmp_path / f"{case.value}.jsonl" for case in PocqiJudgingCase
    }
    common = {
        "question_id": "question-1",
        "question_text": "What is the recommended treatment?",
        "specialty": "Cardiology",
        "responses": _responses(),
        "judge_model": "openai/gpt-test",
        "model_caller": caller,
        "output_paths": output_paths,
    }
    judge_pocqi_responses(
        **common,
        settings=PocqiJudgingSettings(
            experiment_id="experiment-1",
            run_id="run-1",
            retry_delay_seconds=0,
        ),
    )
    forced = judge_pocqi_responses(
        **common,
        settings=PocqiJudgingSettings(
            experiment_id="experiment-1",
            run_id="run-2",
            retry_delay_seconds=0,
            force=True,
        ),
    )

    assert len(caller.calls) == 6
    assert all(record.attempt == 2 for record in forced.values())
    assert all(
        len(path.read_text(encoding="utf-8").splitlines()) == 2
        for path in output_paths.values()
    )


def test_failed_attempt_is_saved_and_retried(tmp_path: Path) -> None:
    successful_caller = FakeJudgeCaller()
    calls = 0

    def flaky_caller(model: str, input: str, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient provider failure")
        return successful_caller(model, input, **kwargs)

    output_paths = {
        case: tmp_path / f"{case.value}.jsonl" for case in PocqiJudgingCase
    }
    records = judge_pocqi_responses(
        question_id="question-1",
        question_text="What is the recommended treatment?",
        specialty="Cardiology",
        responses=_responses(),
        judge_model="openai/gpt-test",
        settings=PocqiJudgingSettings(
            experiment_id="experiment-1",
            run_id="run-1",
            retries=1,
            retry_delay_seconds=0,
        ),
        model_caller=flaky_caller,
        output_paths=output_paths,
    )

    direct_lines = output_paths[
        PocqiJudgingCase.DIRECT_RANKING
    ].read_text(encoding="utf-8").splitlines()
    direct_attempts = [
        PocqiJudgmentRecord.model_validate_json(line) for line in direct_lines
    ]
    assert calls == 4
    assert [record.status for record in direct_attempts] == [
        PocqiJudgmentStatus.FAILED,
        PocqiJudgmentStatus.SUCCEEDED,
    ]
    assert [record.attempt for record in direct_attempts] == [1, 2]
    assert all(record.status is PocqiJudgmentStatus.SUCCEEDED for record in records.values())
