"""Serializable output records for the Real-POCQi generation experiment.

Each :class:`RealPocqiOutput` represents one attempt by one generator model to
answer one Real-POCQi question. Records are intended to be written append-only,
one JSON object per line.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Self


class GenerationStatus(str, Enum):
    """Terminal state of a generation attempt."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RealPocqiOutput:
    """One model-generation attempt for one Real-POCQi question.

    ``generation_key`` identifies the logical question/model generation and is
    shared by retries. ``generation_id`` uniquely identifies this particular
    attempt. This distinction makes retries auditable while allowing consumers
    to select the latest successful attempt for each logical generation.
    """

    experiment_id: str
    run_id: str
    generation_key: str
    generation_id: str
    attempt: int
    status: GenerationStatus
    created_at: str

    question_id: str
    question_text: str
    specialty: str

    generator_family: str
    generator_model: str
    prompt_template_id: str
    system_prompt: str
    user_prompt: str

    response_text: str | None = None
    generator_model_version: str | None = None
    finish_reason: str | None = None

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
        if not isinstance(self.status, GenerationStatus):
            raise TypeError("status must be a GenerationStatus")

        required_strings = {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "generation_key": self.generation_key,
            "generation_id": self.generation_id,
            "created_at": self.created_at,
            "question_id": self.question_id,
            "question_text": self.question_text,
            "specialty": self.specialty,
            "generator_family": self.generator_family,
            "generator_model": self.generator_model,
            "prompt_template_id": self.prompt_template_id,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
        }
        for field_name, value in required_strings.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

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

        if self.status is GenerationStatus.SUCCEEDED:
            if self.response_text is None:
                raise ValueError("a succeeded generation requires response_text")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("a succeeded generation cannot contain an error")
        elif self.status is GenerationStatus.FAILED:
            if not self.error_type and not self.error_message:
                raise ValueError("a failed generation requires error details")

    @staticmethod
    def build_generation_key(
        experiment_id: str,
        question_id: str,
        generator_model: str,
    ) -> str:
        """Build the stable key shared by all attempts of one generation."""

        return "__".join((experiment_id, question_id, generator_model))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""

        record = asdict(self)
        record["status"] = self.status.value
        return record

    def to_json(self) -> str:
        """Serialize the record as one compact JSONL-compatible line."""

        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Self:
        """Validate and construct a record loaded from JSON-compatible data."""

        values = dict(record)
        values["status"] = GenerationStatus(values["status"])
        return cls(**values)

    @classmethod
    def from_json(cls, line: str) -> Self:
        """Deserialize and validate one JSONL line."""

        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("a Real-POCQi output line must contain a JSON object")
        return cls.from_dict(record)
