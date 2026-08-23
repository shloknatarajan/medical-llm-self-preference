"""Serializable pairwise judgments for the Real-POCQi experiment.

Each :class:`RealPocqiJudgment` represents one attempt by one judge model to
evaluate a blinded A/B pair for one question. Records are designed for an
append-only JSONL file, including failed attempts and retries.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from numbers import Real
from typing import Any, Self


class JudgmentStatus(str, Enum):
    """Terminal state of a judgment attempt."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class PairwisePreference(str, Enum):
    """Judge's decision, made independently from the numeric scores."""

    RESPONSE_A = "response_a"
    RESPONSE_B = "response_b"
    TIE = "tie"


@dataclass(frozen=True, slots=True)
class ClinicalScores:
    """The five clinical evaluation scores for one response."""

    faithfulness: float
    completeness: float
    safety: float
    clarity: float
    conciseness: float

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field_name} must be numeric")
            if not 0 <= value <= 5:
                raise ValueError(f"{field_name} must be between 0 and 5")

    @property
    def overall(self) -> float:
        """Mean score used by the score-level self-preference analysis."""

        return sum(asdict(self).values()) / 5

    def to_dict(self) -> dict[str, float]:
        """Return criterion scores plus their derived overall mean."""

        return {**asdict(self), "overall": self.overall}

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Self:
        """Construct scores and verify a serialized overall value, if present."""

        values = dict(record)
        serialized_overall = values.pop("overall", None)
        scores = cls(**values)
        if serialized_overall is not None and not math.isclose(
            serialized_overall,
            scores.overall,
            abs_tol=1e-9,
        ):
            raise ValueError("overall must equal the mean of the five scores")
        return scores


@dataclass(frozen=True, slots=True)
class RealPocqiJudgment:
    """One judge-model attempt to evaluate one blinded Real-POCQi pair.

    ``comparison_key`` is shared by the two judges that see the same A/B
    assignment. ``judgment_key`` additionally identifies the judge and is
    shared by its retries. ``judgment_id`` uniquely identifies this attempt.
    """

    experiment_id: str
    run_id: str
    comparison_key: str
    judgment_key: str
    judgment_id: str
    attempt: int
    status: JudgmentStatus
    created_at: str

    question_id: str
    question_text: str
    specialty: str

    response_a_generation_id: str
    response_a_generator_family: str
    response_a_generator_model: str
    response_b_generation_id: str
    response_b_generator_family: str
    response_b_generator_model: str

    judge_family: str
    judge_model: str
    prompt_template_id: str
    system_prompt: str
    user_prompt: str

    response_a_scores: ClinicalScores | None = None
    response_b_scores: ClinicalScores | None = None
    preference: PairwisePreference | None = None
    preference_reasoning: str | None = None

    judge_model_version: str | None = None
    judge_response_text: str | None = None
    finish_reason: str | None = None

    identity_blinded: bool = True
    position_seed: int | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    seed: int | None = None

    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    provider_request_id: str | None = None

    error_type: str | None = None
    error_message: str | None = None
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not isinstance(self.status, JudgmentStatus):
            raise TypeError("status must be a JudgmentStatus")
        if self.preference is not None and not isinstance(
            self.preference,
            PairwisePreference,
        ):
            raise TypeError("preference must be a PairwisePreference")
        if not isinstance(self.identity_blinded, bool):
            raise TypeError("identity_blinded must be a bool")

        required_strings = {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "comparison_key": self.comparison_key,
            "judgment_key": self.judgment_key,
            "judgment_id": self.judgment_id,
            "created_at": self.created_at,
            "question_id": self.question_id,
            "question_text": self.question_text,
            "specialty": self.specialty,
            "response_a_generation_id": self.response_a_generation_id,
            "response_a_generator_family": self.response_a_generator_family,
            "response_a_generator_model": self.response_a_generator_model,
            "response_b_generation_id": self.response_b_generation_id,
            "response_b_generator_family": self.response_b_generator_family,
            "response_b_generator_model": self.response_b_generator_model,
            "judge_family": self.judge_family,
            "judge_model": self.judge_model,
            "prompt_template_id": self.prompt_template_id,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
        }
        for field_name, value in required_strings.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        if self.response_a_generation_id == self.response_b_generation_id:
            raise ValueError("response A and response B must be different generations")
        if self.attempt < 1:
            raise ValueError("attempt must be at least 1")

        try:
            timestamp = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("created_at must be an ISO-8601 timestamp") from exc
        if timestamp.tzinfo is None:
            raise ValueError("created_at must include a timezone")

        for field_name in (
            "max_output_tokens",
            "input_tokens",
            "output_tokens",
            "latency_ms",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} cannot be negative")

        if self.status is JudgmentStatus.SUCCEEDED:
            if self.response_a_scores is None or self.response_b_scores is None:
                raise ValueError("a succeeded judgment requires scores for A and B")
            if self.preference is None:
                raise ValueError("a succeeded judgment requires a preference")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("a succeeded judgment cannot contain an error")
        elif self.status is JudgmentStatus.FAILED:
            if not self.error_type and not self.error_message:
                raise ValueError("a failed judgment requires error details")
            if (
                self.response_a_scores is not None
                or self.response_b_scores is not None
                or self.preference is not None
            ):
                raise ValueError("a failed judgment cannot contain parsed outcomes")

    @staticmethod
    def build_comparison_key(
        experiment_id: str,
        question_id: str,
        response_a_generation_id: str,
        response_b_generation_id: str,
    ) -> str:
        """Build the key shared by judges viewing the same A/B assignment."""

        return "__".join(
            (
                experiment_id,
                question_id,
                response_a_generation_id,
                response_b_generation_id,
            )
        )

    @staticmethod
    def build_judgment_key(comparison_key: str, judge_model: str) -> str:
        """Build the stable key shared by retries from the same judge."""

        return "__".join((comparison_key, judge_model))

    @property
    def preferred_generation_id(self) -> str | None:
        """Resolve the preference to a generation ID; ties have no winner."""

        if self.preference is PairwisePreference.RESPONSE_A:
            return self.response_a_generation_id
        if self.preference is PairwisePreference.RESPONSE_B:
            return self.response_b_generation_id
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary with database-friendly fields."""

        record = asdict(self)
        record["status"] = self.status.value
        record["preference"] = self.preference.value if self.preference else None
        record["response_a_scores"] = (
            self.response_a_scores.to_dict() if self.response_a_scores else None
        )
        record["response_b_scores"] = (
            self.response_b_scores.to_dict() if self.response_b_scores else None
        )
        record["preferred_generation_id"] = self.preferred_generation_id
        return record

    def to_json(self) -> str:
        """Serialize the record as one compact JSONL-compatible line."""

        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Self:
        """Validate and construct a judgment from JSON-compatible data."""

        values = dict(record)
        values.pop("preferred_generation_id", None)
        values["status"] = JudgmentStatus(values["status"])
        if values.get("preference") is not None:
            values["preference"] = PairwisePreference(values["preference"])
        if values.get("response_a_scores") is not None:
            values["response_a_scores"] = ClinicalScores.from_dict(
                values["response_a_scores"]
            )
        if values.get("response_b_scores") is not None:
            values["response_b_scores"] = ClinicalScores.from_dict(
                values["response_b_scores"]
            )
        return cls(**values)

    @classmethod
    def from_json(cls, line: str) -> Self:
        """Deserialize and validate one JSONL line."""

        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("a Real-POCQi judgment line must contain a JSON object")
        return cls.from_dict(record)
