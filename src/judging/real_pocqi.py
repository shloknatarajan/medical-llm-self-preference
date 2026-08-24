"""Structured results and saved records for Real-POCQi judging."""

import hashlib
import json
import os
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


AbsoluteScore = Annotated[
    float,
    Field(strict=True, ge=0, le=5),
]

DEFAULT_POCQI_JUDGMENTS_DIR = Path("data/real_pcoqi/judgements")
_APPEND_LOCK = threading.Lock()


class RealPocqiScores(BaseModel):
    """Absolute 0--5 scores on the five original Real-POCQi axes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    accuracy: AbsoluteScore = Field(
        description="How factually accurate the response is.",
    )
    clinical_utility: AbsoluteScore = Field(
        description="How useful the response is for providing high-quality clinical care.",
    )
    source_quality: AbsoluteScore = Field(
        description="How authoritative the source material supporting the response is.",
    )
    verifiability: AbsoluteScore = Field(
        description="How easy the response is to verify.",
    )
    completeness: AbsoluteScore = Field(
        description="How completely the response addresses the question.",
    )


class PocqiJudgingCase(str, Enum):
    """Supported Real-POCQi judging procedures."""

    DIRECT_RANKING = "direct_ranking"
    RUBRIC_SUM_RANKING = "rubric_sum_ranking"
    RUBRIC_AND_MODEL_RANKING = "rubric_and_model_ranking"


class PocqiJudgmentStatus(str, Enum):
    """Terminal state of a saved judging attempt."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


POCQI_JUDGMENT_PATHS: dict[PocqiJudgingCase, Path] = {
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


class PocqiResponseCandidate(BaseModel):
    """Reference to one generated response shown to the judge.

    Candidate list order is the presentation order seen by the judge.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    response_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    generator_family: str = Field(min_length=1)
    generator_model: str = Field(min_length=1)


class ResponseRanking(BaseModel):
    """A strict total ordering of response IDs from best to worst."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    response_ids: list[str] = Field(
        min_length=2,
        description="Response IDs ordered from best-performing to worst-performing.",
    )

    @model_validator(mode="after")
    def validate_unique_response_ids(self) -> "ResponseRanking":
        if any(not response_id for response_id in self.response_ids):
            raise ValueError("ranking response IDs must be non-empty")
        if len(set(self.response_ids)) != len(self.response_ids):
            raise ValueError("ranking response IDs must be unique")
        return self


class ScoredPocqiResponse(BaseModel):
    """Absolute rubric scores for one response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    response_id: str = Field(min_length=1)
    scores: RealPocqiScores

    @computed_field(return_type=float)
    @property
    def score_sum(self) -> float:
        """Unweighted sum across the five Real-POCQi axes."""

        return sum(self.scores.model_dump().values())


def _score_sum_ranking(
    scored_responses: list[ScoredPocqiResponse],
) -> ResponseRanking:
    """Rank by descending rubric sum, breaking ties by response ID."""

    ordered = sorted(
        scored_responses,
        key=lambda scored: (-scored.score_sum, scored.response_id),
    )
    return ResponseRanking(response_ids=[scored.response_id for scored in ordered])


class DirectRankingOutput(BaseModel):
    """Provider-facing output for direct ranking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ranking: ResponseRanking


class RubricScoringOutput(BaseModel):
    """Provider-facing output containing rubric scores only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scored_responses: list[ScoredPocqiResponse] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_unique_scored_responses(self) -> "RubricScoringOutput":
        response_ids = [item.response_id for item in self.scored_responses]
        if len(set(response_ids)) != len(response_ids):
            raise ValueError("scored response IDs must be unique")
        return self


class RubricAndModelRankingOutput(RubricScoringOutput):
    """Provider-facing output containing scores and a holistic ranking."""

    model_ranking: ResponseRanking

    @model_validator(mode="after")
    def validate_output_response_ids(self) -> "RubricAndModelRankingOutput":
        scored_ids = [item.response_id for item in self.scored_responses]
        if set(self.model_ranking.response_ids) != set(scored_ids):
            raise ValueError(
                "model ranking must contain exactly the scored response IDs"
            )
        return self


class DirectRankingResult(BaseModel):
    """Case 1: the judge directly ranks responses without a rubric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case: Literal[PocqiJudgingCase.DIRECT_RANKING] = (
        PocqiJudgingCase.DIRECT_RANKING
    )
    ranking: ResponseRanking


class RubricSumRankingResult(BaseModel):
    """Case 2: rubric scores determine the ranking mechanically."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case: Literal[PocqiJudgingCase.RUBRIC_SUM_RANKING] = (
        PocqiJudgingCase.RUBRIC_SUM_RANKING
    )
    scored_responses: list[ScoredPocqiResponse] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_unique_scored_responses(self) -> "RubricSumRankingResult":
        response_ids = [item.response_id for item in self.scored_responses]
        if len(set(response_ids)) != len(response_ids):
            raise ValueError("scored response IDs must be unique")
        return self

    @computed_field(return_type=ResponseRanking)
    @property
    def ranking(self) -> ResponseRanking:
        """Deterministic ranking derived from rubric-score sums."""

        return _score_sum_ranking(self.scored_responses)


class RubricAndModelRankingResult(BaseModel):
    """Case 3: the judge supplies rubric scores and an independent ranking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case: Literal[PocqiJudgingCase.RUBRIC_AND_MODEL_RANKING] = (
        PocqiJudgingCase.RUBRIC_AND_MODEL_RANKING
    )
    scored_responses: list[ScoredPocqiResponse] = Field(min_length=2)
    model_ranking: ResponseRanking

    @model_validator(mode="after")
    def validate_result_response_ids(self) -> "RubricAndModelRankingResult":
        scored_ids = [item.response_id for item in self.scored_responses]
        if len(set(scored_ids)) != len(scored_ids):
            raise ValueError("scored response IDs must be unique")
        if set(self.model_ranking.response_ids) != set(scored_ids):
            raise ValueError(
                "model ranking must contain exactly the scored response IDs"
            )
        return self

    @computed_field(return_type=ResponseRanking)
    @property
    def score_sum_ranking(self) -> ResponseRanking:
        """Mechanical ranking retained for comparison with the model ranking."""

        return _score_sum_ranking(self.scored_responses)


PocqiJudgmentResult = Annotated[
    DirectRankingResult | RubricSumRankingResult | RubricAndModelRankingResult,
    Field(discriminator="case"),
]


class PocqiJudgmentRecord(BaseModel):
    """Append-only saved record for one Real-POCQi judging attempt.

    Candidate provenance is retained here for analysis, but it is not part of
    the identity-blinded prompt sent to the judge.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0", "1.1"] = "1.1"
    experiment_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    judgment_key: str | None = None
    judgment_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    judging_case: PocqiJudgingCase
    status: PocqiJudgmentStatus
    created_at: datetime

    question_id: str = Field(min_length=1)
    question_text: str = Field(min_length=1)
    specialty: str = Field(min_length=1)
    candidates: list[PocqiResponseCandidate] = Field(min_length=2)

    judge_family: str = Field(min_length=1)
    judge_model: str = Field(min_length=1)
    prompt_template_id: str = Field(min_length=1)
    system_prompt: str
    user_prompt: str = Field(min_length=1)

    result: PocqiJudgmentResult | None = None
    judge_response_text: str | None = None
    finish_reason: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    seed: int | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: int | None = Field(default=None, ge=0)
    provider_request_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_record(self) -> "PocqiJudgmentRecord":
        candidate_ids = [candidate.response_id for candidate in self.candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate response IDs must be unique")

        if self.status is PocqiJudgmentStatus.SUCCEEDED:
            if self.result is None:
                raise ValueError("a succeeded judgment requires a result")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("a succeeded judgment cannot contain an error")
            if self.result.case != self.judging_case:
                raise ValueError("result case must match the record judging case")
            result_ids = self._result_response_ids(self.result)
            if set(result_ids) != set(candidate_ids):
                raise ValueError(
                    "result must contain exactly the candidate response IDs"
                )
        elif self.result is not None:
            raise ValueError("a failed judgment cannot contain a result")

        return self

    @staticmethod
    def _result_response_ids(result: PocqiJudgmentResult) -> list[str]:
        if isinstance(result, DirectRankingResult):
            return result.ranking.response_ids
        return [item.response_id for item in result.scored_responses]

    def resolved_rankings(self) -> dict[str, list[str]]:
        """Return stored rankings resolved from blind IDs to generator models."""

        if self.result is None:
            return {}
        model_by_response_id = {
            candidate.response_id: candidate.generator_model
            for candidate in self.candidates
        }

        def resolve(ranking: ResponseRanking) -> list[str]:
            return [model_by_response_id[item] for item in ranking.response_ids]

        if isinstance(self.result, DirectRankingResult):
            return {"direct_ranking": resolve(self.result.ranking)}
        if isinstance(self.result, RubricSumRankingResult):
            return {"score_sum_ranking": resolve(self.result.ranking)}
        return {
            "model_ranking": resolve(self.result.model_ranking),
            "score_sum_ranking": resolve(self.result.score_sum_ranking),
        }

    @staticmethod
    def build_judgment_key(
        *,
        experiment_id: str,
        question_id: str,
        judge_model: str,
        judging_case: PocqiJudgingCase,
        prompt_template_id: str,
        presentation_seed: int | None,
        generation_ids: list[str],
    ) -> str:
        """Build the stable key shared by retries of one logical judgment."""

        payload = json.dumps(
            {
                "experiment_id": experiment_id,
                "question_id": question_id,
                "judge_model": judge_model,
                "judging_case": judging_case.value,
                "prompt_template_id": prompt_template_id,
                "presentation_seed": presentation_seed,
                # Candidate identity is set-based for resumability. Presentation
                # order is separately and deterministically controlled by the seed.
                "generation_ids": sorted(generation_ids),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return "__".join(
            (experiment_id, question_id, judge_model, judging_case.value, digest)
        )

    def resolved_judgment_key(self) -> str:
        """Return the stored key or derive it for legacy schema-1.0 records."""

        if self.judgment_key:
            return self.judgment_key
        return self.build_judgment_key(
            experiment_id=self.experiment_id,
            question_id=self.question_id,
            judge_model=self.judge_model,
            judging_case=self.judging_case,
            prompt_template_id=self.prompt_template_id,
            presentation_seed=self.seed,
            generation_ids=[
                candidate.generation_id for candidate in self.candidates
            ],
        )


def append_pocqi_judgment(
    record: PocqiJudgmentRecord,
    output_path: str | Path | None = None,
) -> Path:
    """Append a record to the JSONL file for its judging case.

    Computed score sums and deterministic rankings are reconstructed when a
    record is loaded, so they are not redundantly persisted on disk.
    """

    path = Path(output_path) if output_path is not None else POCQI_JUDGMENT_PATHS[
        record.judging_case
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    line = record.model_dump_json(exclude_computed_fields=True) + "\n"
    with _APPEND_LOCK, path.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
    return path
