"""Tests for Real-POCQi judging schemas and persistence."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from judging import (
    DEFAULT_POCQI_JUDGMENTS_DIR,
    DirectRankingResult,
    POCQI_JUDGMENT_PATHS,
    PocqiJudgingCase,
    PocqiJudgmentRecord,
    PocqiJudgmentStatus,
    PocqiResponseCandidate,
    RealPocqiScores,
    ResponseRanking,
    RubricAndModelRankingResult,
    RubricSumRankingResult,
    ScoredPocqiResponse,
    append_pocqi_judgment,
)


def test_real_pocqi_scores_accept_values_in_closed_interval() -> None:
    scores = RealPocqiScores(
        accuracy=0,
        clinical_utility=1.5,
        source_quality=3,
        verifiability=4.5,
        completeness=5,
    )

    assert scores.model_dump() == {
        "accuracy": 0.0,
        "clinical_utility": 1.5,
        "source_quality": 3.0,
        "verifiability": 4.5,
        "completeness": 5.0,
    }


@pytest.mark.parametrize("invalid_score", [-0.01, 5.01])
def test_real_pocqi_scores_reject_out_of_range_values(invalid_score: float) -> None:
    with pytest.raises(ValidationError):
        RealPocqiScores(
            accuracy=invalid_score,
            clinical_utility=3,
            source_quality=3,
            verifiability=3,
            completeness=3,
        )


def test_real_pocqi_scores_require_every_axis() -> None:
    with pytest.raises(ValidationError):
        RealPocqiScores(
            accuracy=3,
            clinical_utility=3,
            source_quality=3,
            verifiability=3,
        )


def test_real_pocqi_scores_reject_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RealPocqiScores(
            accuracy=3,
            clinical_utility=3,
            source_quality=3,
            verifiability=3,
            completeness=3,
            overall=3,
        )


def _scores(value: float) -> RealPocqiScores:
    return RealPocqiScores(
        accuracy=value,
        clinical_utility=value,
        source_quality=value,
        verifiability=value,
        completeness=value,
    )


def _candidates() -> list[PocqiResponseCandidate]:
    return [
        PocqiResponseCandidate(
            response_id=response_id,
            generation_id=f"generation-{response_id}",
            generator_family=f"family-{response_id}",
            generator_model=f"model-{response_id}",
        )
        for response_id in ("response-a", "response-b", "response-c")
    ]


def _record(result) -> PocqiJudgmentRecord:
    return PocqiJudgmentRecord(
        experiment_id="experiment-1",
        run_id="run-1",
        judgment_id="judgment-1",
        attempt=1,
        judging_case=result.case,
        status=PocqiJudgmentStatus.SUCCEEDED,
        created_at=datetime(2026, 8, 24, tzinfo=UTC),
        question_id="question-1",
        question_text="What is the appropriate treatment?",
        specialty="Medicine",
        candidates=_candidates(),
        judge_family="judge-family",
        judge_model="judge-model",
        prompt_template_id="prompt-v1",
        system_prompt="",
        user_prompt="Rank the responses.",
        result=result,
    )


def test_direct_ranking_result_round_trips_in_saved_record() -> None:
    result = DirectRankingResult(
        ranking=ResponseRanking(
            response_ids=["response-b", "response-a", "response-c"]
        )
    )

    record = _record(result)
    restored = PocqiJudgmentRecord.model_validate_json(record.model_dump_json())

    assert isinstance(restored.result, DirectRankingResult)
    assert restored.result.ranking.response_ids == [
        "response-b",
        "response-a",
        "response-c",
    ]


def test_rubric_sum_result_computes_scores_and_deterministic_ranking() -> None:
    result = RubricSumRankingResult(
        scored_responses=[
            ScoredPocqiResponse(response_id="response-c", scores=_scores(4)),
            ScoredPocqiResponse(response_id="response-b", scores=_scores(4)),
            ScoredPocqiResponse(response_id="response-a", scores=_scores(3)),
        ]
    )

    assert result.scored_responses[0].score_sum == 20
    assert result.ranking.response_ids == [
        "response-b",
        "response-c",
        "response-a",
    ]
    assert result.model_dump()["ranking"]["response_ids"] == [
        "response-b",
        "response-c",
        "response-a",
    ]

    record = _record(result)
    serialized = record.model_dump_json(exclude_computed_fields=True)
    restored = PocqiJudgmentRecord.model_validate_json(serialized)
    assert restored.result.ranking.response_ids == [
        "response-b",
        "response-c",
        "response-a",
    ]


def test_rubric_and_model_ranking_keeps_independent_model_order() -> None:
    result = RubricAndModelRankingResult(
        scored_responses=[
            ScoredPocqiResponse(response_id="response-a", scores=_scores(5)),
            ScoredPocqiResponse(response_id="response-b", scores=_scores(4)),
            ScoredPocqiResponse(response_id="response-c", scores=_scores(3)),
        ],
        model_ranking=ResponseRanking(
            response_ids=["response-c", "response-b", "response-a"]
        ),
    )

    assert result.model_ranking.response_ids == [
        "response-c",
        "response-b",
        "response-a",
    ]
    assert result.score_sum_ranking.response_ids == [
        "response-a",
        "response-b",
        "response-c",
    ]


def test_saved_record_requires_result_to_cover_every_candidate() -> None:
    with pytest.raises(ValidationError, match="exactly the candidate response IDs"):
        _record(
            DirectRankingResult(
                ranking=ResponseRanking(
                    response_ids=["response-a", "response-b"]
                )
            )
        )


def test_append_pocqi_judgment_writes_jsonl(tmp_path) -> None:
    record = _record(
        DirectRankingResult(
            ranking=ResponseRanking(
                response_ids=["response-a", "response-b", "response-c"]
            )
        )
    )
    output_path = tmp_path / "judgements" / "judgments.jsonl"

    written_path = append_pocqi_judgment(record, output_path)
    append_pocqi_judgment(record, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert written_path == output_path
    assert len(lines) == 2
    assert PocqiJudgmentRecord.model_validate_json(lines[0]) == record


def test_default_judgment_directory_matches_experiment_layout() -> None:
    assert DEFAULT_POCQI_JUDGMENTS_DIR.as_posix() == "data/real_pcoqi/judgements"


def test_each_judging_case_has_a_separate_default_jsonl() -> None:
    assert POCQI_JUDGMENT_PATHS == {
        PocqiJudgingCase.DIRECT_RANKING: (
            DEFAULT_POCQI_JUDGMENTS_DIR / "direct_ranking.jsonl"
        ),
        PocqiJudgingCase.RUBRIC_SUM_RANKING: (
            DEFAULT_POCQI_JUDGMENTS_DIR / "rubric_sum_ranking.jsonl"
        ),
        PocqiJudgingCase.RUBRIC_AND_MODEL_RANKING: (
            DEFAULT_POCQI_JUDGMENTS_DIR / "rubric_and_model_ranking.jsonl"
        ),
    }


def test_record_rejects_a_result_from_a_different_judging_case() -> None:
    result = DirectRankingResult(
        ranking=ResponseRanking(
            response_ids=["response-a", "response-b", "response-c"]
        )
    )
    values = _record(result).model_dump(exclude_computed_fields=True)
    values["judging_case"] = PocqiJudgingCase.RUBRIC_SUM_RANKING

    with pytest.raises(ValidationError, match="result case must match"):
        PocqiJudgmentRecord.model_validate(values)
