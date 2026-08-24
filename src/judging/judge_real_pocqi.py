"""Run the three independent Real-POCQi judging conditions."""

from __future__ import annotations

import random
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inference import ModalConfig, ModelResponse, Provider, call_model, resolve_provider

from .real_pocqi import (
    POCQI_JUDGMENT_PATHS,
    DirectRankingOutput,
    DirectRankingResult,
    PocqiJudgingCase,
    PocqiJudgmentRecord,
    PocqiJudgmentResult,
    PocqiJudgmentStatus,
    PocqiResponseCandidate,
    RubricAndModelRankingOutput,
    RubricAndModelRankingResult,
    RubricScoringOutput,
    RubricSumRankingResult,
    append_pocqi_judgment,
)


DIRECT_RANKING_PROMPT_ID = "pocqi_direct_quality_ranking_v2"
RUBRIC_SUM_PROMPT_ID = "pocqi_rubric_sum_ranking_v1"
RUBRIC_AND_RANKING_PROMPT_ID = "pocqi_rubric_and_model_ranking_v1"
IDENTITY_REVEALED_DIRECT_RANKING_PROMPT_ID = (
    "pocqi_identity_revealed_direct_quality_ranking_v1"
)
IDENTITY_REVEALED_RUBRIC_SUM_PROMPT_ID = (
    "pocqi_identity_revealed_rubric_sum_ranking_v1"
)
IDENTITY_REVEALED_RUBRIC_AND_RANKING_PROMPT_ID = (
    "pocqi_identity_revealed_rubric_and_model_ranking_v1"
)

DIRECT_RANKING_SYSTEM_PROMPT = (
    "You are evaluating the overall quality of several responses to the same "
    "clinical question."
)

RUBRIC_SYSTEM_PROMPT = """You are an expert US physician evaluating responses to a clinical question asked by a physician.

Give each response an absolute score from 0 to 5 on each axis, where 0 is the lowest possible performance and 5 is the highest:
- accuracy: How factually accurate is the response?
- clinical_utility: How useful is the response for providing high-quality clinical care?
- source_quality: How authoritative is the source material supporting the response?
- verifiability: How easy is the response to verify?
- completeness: How completely does the response address the question?

Score every response independently. Use only the information in the clinical question and candidate responses. Do not infer or identify which model produced a response."""

IDENTITY_REVEALED_DIRECT_RANKING_SYSTEM_PROMPT = (
    DIRECT_RANKING_SYSTEM_PROMPT
    + " The generator model is explicitly supplied for each candidate."
)
IDENTITY_REVEALED_RUBRIC_SYSTEM_PROMPT = RUBRIC_SYSTEM_PROMPT.replace(
    "Do not infer or identify which model produced a response.",
    "The generator model is explicitly supplied for each candidate.",
)

DIRECT_RANKING_INSTRUCTION = (
    "Without using a predefined rubric, rank all candidate responses from "
    "highest overall quality to lowest overall quality. Return every response "
    "ID exactly once."
)
RUBRIC_SUM_INSTRUCTION = (
    "Score every candidate response on all five rubric axes. Do not provide a "
    "ranking; it will be calculated deterministically from the score sums."
)
RUBRIC_AND_RANKING_INSTRUCTION = (
    "First score every candidate response on all five rubric axes. Then rank "
    "all responses from best-performing to worst-performing using your overall "
    "clinical judgment. The ranking does not need to follow the score sums. "
    "Return every response ID exactly once."
)

ModelCaller = Callable[..., ModelResponse[Any]]
OutputType = type[
    DirectRankingOutput | RubricScoringOutput | RubricAndModelRankingOutput
]


@dataclass(frozen=True, slots=True)
class PocqiResponseInput:
    """One generated response supplied to the judging function."""

    generation_id: str
    generator_family: str
    generator_model: str
    response_text: str

    def __post_init__(self) -> None:
        for field_name in (
            "generation_id",
            "generator_family",
            "generator_model",
            "response_text",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class PocqiJudgingSettings:
    """Settings shared by the three judging calls for one candidate set."""

    experiment_id: str
    run_id: str
    presentation_seed: int = 42
    temperature: float | None = None
    max_output_tokens: int = 4096
    retries: int = 2
    retry_delay_seconds: float = 2.0
    force: bool = False
    modal_config: ModalConfig | None = None

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must be non-empty")
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.retries < 0:
            raise ValueError("retries cannot be negative")
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds cannot be negative")


@dataclass(frozen=True, slots=True)
class PocqiResumeState:
    """Terminal state recovered from one condition-specific JSONL file."""

    succeeded_by_key: dict[str, PocqiJudgmentRecord]
    highest_attempt_by_key: dict[str, int]


class PocqiResumeTracker:
    """Thread-safe in-memory resume index for a batch run."""

    def __init__(self, paths: Sequence[str | Path]) -> None:
        self._lock = threading.Lock()
        self._states = {
            Path(path): load_pocqi_resume_state(path) for path in paths
        }

    def lookup(
        self,
        path: str | Path,
        judgment_key: str,
    ) -> tuple[PocqiJudgmentRecord | None, int]:
        with self._lock:
            state = self._states[Path(path)]
            return (
                state.succeeded_by_key.get(judgment_key),
                state.highest_attempt_by_key.get(judgment_key, 0),
            )

    def update(self, path: str | Path, record: PocqiJudgmentRecord) -> None:
        key = record.resolved_judgment_key()
        with self._lock:
            state = self._states[Path(path)]
            state.highest_attempt_by_key[key] = max(
                record.attempt,
                state.highest_attempt_by_key.get(key, 0),
            )
            if record.status is PocqiJudgmentStatus.SUCCEEDED:
                state.succeeded_by_key[key] = record


@dataclass(frozen=True, slots=True)
class JudgingCondition:
    case: PocqiJudgingCase
    prompt_template_id: str
    system_prompt: str
    instruction: str
    output_type: OutputType


UserPromptBuilder = Callable[
    [str, str, Sequence[PocqiResponseCandidate], Mapping[str, str], str],
    str,
]


@dataclass(frozen=True, slots=True)
class JudgingTaskProfile:
    """Task-specific prompts layered over the shared blinded judge."""

    conditions: tuple[JudgingCondition, ...]
    user_prompt_builder: UserPromptBuilder
    identity_blinded: bool = True


JUDGING_CONDITIONS = (
    JudgingCondition(
        case=PocqiJudgingCase.DIRECT_RANKING,
        prompt_template_id=DIRECT_RANKING_PROMPT_ID,
        system_prompt=DIRECT_RANKING_SYSTEM_PROMPT,
        instruction=DIRECT_RANKING_INSTRUCTION,
        output_type=DirectRankingOutput,
    ),
    JudgingCondition(
        case=PocqiJudgingCase.RUBRIC_SUM_RANKING,
        prompt_template_id=RUBRIC_SUM_PROMPT_ID,
        system_prompt=RUBRIC_SYSTEM_PROMPT,
        instruction=RUBRIC_SUM_INSTRUCTION,
        output_type=RubricScoringOutput,
    ),
    JudgingCondition(
        case=PocqiJudgingCase.RUBRIC_AND_MODEL_RANKING,
        prompt_template_id=RUBRIC_AND_RANKING_PROMPT_ID,
        system_prompt=RUBRIC_SYSTEM_PROMPT,
        instruction=RUBRIC_AND_RANKING_INSTRUCTION,
        output_type=RubricAndModelRankingOutput,
    ),
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _saved_result(
    case: PocqiJudgingCase,
    output: DirectRankingOutput | RubricScoringOutput | RubricAndModelRankingOutput,
) -> PocqiJudgmentResult:
    if case is PocqiJudgingCase.DIRECT_RANKING:
        assert isinstance(output, DirectRankingOutput)
        return DirectRankingResult(ranking=output.ranking)
    if case is PocqiJudgingCase.RUBRIC_SUM_RANKING:
        assert isinstance(output, RubricScoringOutput)
        return RubricSumRankingResult(scored_responses=output.scored_responses)
    assert isinstance(output, RubricAndModelRankingOutput)
    return RubricAndModelRankingResult(
        scored_responses=output.scored_responses,
        model_ranking=output.model_ranking,
    )


def _judge_family(model: str) -> str:
    native_model = model.partition("/")[2] or model
    lowered = native_model.casefold()
    if lowered.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if lowered.startswith("claude-"):
        return "anthropic"
    if lowered.startswith("gemini-"):
        return "google"
    if lowered.startswith("qwen"):
        return "qwen"
    return resolve_provider(model)[0].value


def _ordered_inputs(
    responses: Sequence[PocqiResponseInput],
    *,
    seed: int,
    question_id: str,
    judge_model: str,
) -> list[PocqiResponseInput]:
    # Canonicalize first so caller input order cannot change the blinded slots.
    ordered = sorted(responses, key=lambda response: response.generation_id)
    random.Random(f"{seed}:{question_id}:{judge_model}").shuffle(ordered)
    return ordered


def _label_candidates(
    responses: Sequence[PocqiResponseInput],
) -> tuple[list[PocqiResponseCandidate], dict[str, str]]:
    candidates: list[PocqiResponseCandidate] = []
    texts: dict[str, str] = {}
    for index, response in enumerate(responses, start=1):
        response_id = f"response-{index}"
        candidates.append(
            PocqiResponseCandidate(
                response_id=response_id,
                generation_id=response.generation_id,
                generator_family=response.generator_family,
                generator_model=response.generator_model,
            )
        )
        texts[response_id] = response.response_text
    return candidates, texts


def _user_prompt(
    *,
    question_text: str,
    specialty: str,
    candidates: Sequence[PocqiResponseCandidate],
    response_texts: Mapping[str, str],
    instruction: str,
) -> str:
    rendered_responses = "\n\n".join(
        f'<candidate_response id="{candidate.response_id}">\n'
        f"{response_texts[candidate.response_id]}\n"
        "</candidate_response>"
        for candidate in candidates
    )
    return f"""CLINICAL SPECIALTY:
{specialty}

CLINICAL QUESTION:
{question_text}

CANDIDATE RESPONSES:
{rendered_responses}

TASK:
{instruction}"""


def _pocqi_user_prompt(
    question_text: str,
    specialty: str,
    candidates: Sequence[PocqiResponseCandidate],
    response_texts: Mapping[str, str],
    instruction: str,
) -> str:
    return _user_prompt(
        question_text=question_text,
        specialty=specialty,
        candidates=candidates,
        response_texts=response_texts,
        instruction=instruction,
    )


POCQI_JUDGING_PROFILE = JudgingTaskProfile(
    conditions=JUDGING_CONDITIONS,
    user_prompt_builder=_pocqi_user_prompt,
)


def _identity_revealed_user_prompt(
    question_text: str,
    specialty: str,
    candidates: Sequence[PocqiResponseCandidate],
    response_texts: Mapping[str, str],
    instruction: str,
) -> str:
    rendered_responses = "\n\n".join(
        f'<candidate_response id="{candidate.response_id}" '
        f'generator_model="{candidate.generator_model}">\n'
        f"{response_texts[candidate.response_id]}\n"
        "</candidate_response>"
        for candidate in candidates
    )
    return f"""CLINICAL SPECIALTY:
{specialty}

CLINICAL QUESTION:
{question_text}

CANDIDATE RESPONSES:
{rendered_responses}

TASK:
{instruction}"""


IDENTITY_REVEALED_POCQI_JUDGING_PROFILE = JudgingTaskProfile(
    conditions=tuple(
        JudgingCondition(
            case=condition.case,
            prompt_template_id={
                PocqiJudgingCase.DIRECT_RANKING: (
                    IDENTITY_REVEALED_DIRECT_RANKING_PROMPT_ID
                ),
                PocqiJudgingCase.RUBRIC_SUM_RANKING: (
                    IDENTITY_REVEALED_RUBRIC_SUM_PROMPT_ID
                ),
                PocqiJudgingCase.RUBRIC_AND_MODEL_RANKING: (
                    IDENTITY_REVEALED_RUBRIC_AND_RANKING_PROMPT_ID
                ),
            }[condition.case],
            system_prompt=(
                IDENTITY_REVEALED_DIRECT_RANKING_SYSTEM_PROMPT
                if condition.case is PocqiJudgingCase.DIRECT_RANKING
                else IDENTITY_REVEALED_RUBRIC_SYSTEM_PROMPT
            ),
            instruction=condition.instruction,
            output_type=condition.output_type,
        )
        for condition in JUDGING_CONDITIONS
    ),
    user_prompt_builder=_identity_revealed_user_prompt,
    identity_blinded=False,
)


def _output_path(
    case: PocqiJudgingCase,
    output_paths: Mapping[PocqiJudgingCase, str | Path] | None,
) -> Path:
    if output_paths is None:
        return POCQI_JUDGMENT_PATHS[case]
    try:
        return Path(output_paths[case])
    except KeyError as exc:
        raise ValueError(f"missing output path for judging case {case.value}") from exc


def load_pocqi_resume_state(path: str | Path) -> PocqiResumeState:
    """Recover successes and attempt counters from a judgment JSONL file."""

    judgment_path = Path(path)
    succeeded: dict[str, PocqiJudgmentRecord] = {}
    highest_attempt: dict[str, int] = {}
    if not judgment_path.exists():
        return PocqiResumeState({}, {})

    with judgment_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = PocqiJudgmentRecord.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"invalid judgment at {judgment_path}:{line_number}: {exc}"
                ) from exc
            key = record.resolved_judgment_key()
            highest_attempt[key] = max(
                record.attempt,
                highest_attempt.get(key, 0),
            )
            if record.status is PocqiJudgmentStatus.SUCCEEDED:
                previous = succeeded.get(key)
                if previous is None or record.attempt >= previous.attempt:
                    succeeded[key] = record
    return PocqiResumeState(succeeded, highest_attempt)


def build_judgment_keys(
    *,
    question_id: str,
    responses: Sequence[PocqiResponseInput],
    judge_model: str,
    settings: PocqiJudgingSettings,
    profile: JudgingTaskProfile,
    judging_cases: Sequence[PocqiJudgingCase] = tuple(PocqiJudgingCase),
) -> dict[PocqiJudgingCase, str]:
    """Build stable keys for a task profile without making a model call."""

    _, native_judge_model = resolve_provider(judge_model)
    generation_ids = [response.generation_id for response in responses]
    selected_cases = set(judging_cases)
    return {
        condition.case: PocqiJudgmentRecord.build_judgment_key(
            experiment_id=settings.experiment_id,
            question_id=question_id,
            judge_model=native_judge_model,
            judging_case=condition.case,
            prompt_template_id=condition.prompt_template_id,
            presentation_seed=settings.presentation_seed,
            generation_ids=generation_ids,
        )
        for condition in profile.conditions
        if condition.case in selected_cases
    }


def build_pocqi_judgment_keys(
    *,
    question_id: str,
    responses: Sequence[PocqiResponseInput],
    judge_model: str,
    settings: PocqiJudgingSettings,
    reveal_generator_identities: bool = False,
    judging_cases: Sequence[PocqiJudgingCase] = tuple(PocqiJudgingCase),
) -> dict[PocqiJudgingCase, str]:
    """Build the Real-POCQi stable keys without making a model call."""

    return build_judgment_keys(
        question_id=question_id,
        responses=responses,
        judge_model=judge_model,
        settings=settings,
        profile=(
            IDENTITY_REVEALED_POCQI_JUDGING_PROFILE
            if reveal_generator_identities
            else POCQI_JUDGING_PROFILE
        ),
        judging_cases=judging_cases,
    )


def judge_ranked_responses(
    *,
    question_id: str,
    question_text: str,
    specialty: str,
    responses: Sequence[PocqiResponseInput],
    judge_model: str,
    settings: PocqiJudgingSettings,
    profile: JudgingTaskProfile,
    model_caller: ModelCaller = call_model,
    output_paths: Mapping[PocqiJudgingCase, str | Path] | None = None,
    resume_tracker: PocqiResumeTracker | None = None,
    judging_cases: Sequence[PocqiJudgingCase] = tuple(PocqiJudgingCase),
) -> dict[PocqiJudgingCase, PocqiJudgmentRecord]:
    """Run and save independent judging conditions.

    All calls see the same candidate labels and presentation order. Candidate
    model identities are included only when the supplied profile requests it.
    Each attempt is appended immediately to its condition-specific JSONL file,
    including provider or parsing failures.
    """

    for field_name, value in (
        ("question_id", question_id),
        ("question_text", question_text),
        ("specialty", specialty),
        ("judge_model", judge_model),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
    if len(responses) < 2:
        raise ValueError("at least two responses are required for ranking")
    selected_cases = tuple(judging_cases)
    if not selected_cases:
        raise ValueError("at least one judging case is required")
    if len(set(selected_cases)) != len(selected_cases):
        raise ValueError("judging cases must be unique")
    generation_ids = [response.generation_id for response in responses]
    if len(set(generation_ids)) != len(generation_ids):
        raise ValueError("response generation IDs must be unique")

    provider, native_judge_model = resolve_provider(judge_model)
    ordered = _ordered_inputs(
        responses,
        seed=settings.presentation_seed,
        question_id=question_id,
        judge_model=native_judge_model,
    )
    candidates, response_texts = _label_candidates(ordered)
    judge_family = _judge_family(judge_model)
    records: dict[PocqiJudgingCase, PocqiJudgmentRecord] = {}
    judgment_keys = build_judgment_keys(
        question_id=question_id,
        responses=responses,
        judge_model=judge_model,
        settings=settings,
        profile=profile,
        judging_cases=selected_cases,
    )

    for condition in profile.conditions:
        if condition.case not in selected_cases:
            continue
        path = _output_path(condition.case, output_paths)
        user_prompt = profile.user_prompt_builder(
            question_text,
            specialty,
            candidates,
            response_texts,
            condition.instruction,
        )
        judgment_key = judgment_keys[condition.case]
        if resume_tracker is None:
            resume_state = load_pocqi_resume_state(path)
            succeeded_record = resume_state.succeeded_by_key.get(judgment_key)
            highest_attempt = resume_state.highest_attempt_by_key.get(
                judgment_key,
                0,
            )
        else:
            succeeded_record, highest_attempt = resume_tracker.lookup(
                path,
                judgment_key,
            )
        if not settings.force and succeeded_record is not None:
            records[condition.case] = succeeded_record
            continue

        starting_attempt = highest_attempt + 1
        last_record: PocqiJudgmentRecord | None = None
        for retry_index in range(settings.retries + 1):
            attempt = starting_attempt + retry_index
            started = time.perf_counter()
            common: dict[str, Any] = {
                "experiment_id": settings.experiment_id,
                "run_id": settings.run_id,
                "judgment_key": judgment_key,
                "judgment_id": f"judgment-{uuid.uuid4()}",
                "attempt": attempt,
                "judging_case": condition.case,
                "created_at": _utc_now(),
                "question_id": question_id,
                "question_text": question_text,
                "specialty": specialty,
                "candidates": candidates,
                "judge_family": judge_family,
                "judge_model": native_judge_model,
                "prompt_template_id": condition.prompt_template_id,
                "identity_blinded": profile.identity_blinded,
                "system_prompt": condition.system_prompt,
                "user_prompt": user_prompt,
                "temperature": settings.temperature,
                "max_output_tokens": settings.max_output_tokens,
                "seed": settings.presentation_seed,
            }

            try:
                options: dict[str, Any] = {}
                if settings.temperature is not None:
                    options["temperature"] = settings.temperature
                response = model_caller(
                    judge_model,
                    user_prompt,
                    system=condition.system_prompt,
                    response_format=condition.output_type,
                    max_output_tokens=settings.max_output_tokens,
                    modal_config=(
                        settings.modal_config if provider is Provider.MODAL else None
                    ),
                    **options,
                )
                if not isinstance(response.parsed, condition.output_type):
                    raise TypeError(
                        "judge did not return the expected structured output "
                        f"{condition.output_type.__name__}"
                    )
                saved_result = _saved_result(condition.case, response.parsed)
                last_record = PocqiJudgmentRecord(
                    **common,
                    status=PocqiJudgmentStatus.SUCCEEDED,
                    result=saved_result,
                    judge_response_text=response.text or None,
                    finish_reason=response.finish_reason,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    latency_ms=round((time.perf_counter() - started) * 1000),
                    provider_request_id=response.request_id,
                )
            except Exception as exc:
                last_record = PocqiJudgmentRecord(
                    **common,
                    status=PocqiJudgmentStatus.FAILED,
                    latency_ms=round((time.perf_counter() - started) * 1000),
                    error_type=type(exc).__name__,
                    error_message=str(exc) or repr(exc),
                )

            append_pocqi_judgment(last_record, path)
            if resume_tracker is not None:
                resume_tracker.update(path, last_record)
            if last_record.status is PocqiJudgmentStatus.SUCCEEDED:
                break
            if (
                retry_index < settings.retries
                and settings.retry_delay_seconds > 0
            ):
                time.sleep(settings.retry_delay_seconds * (2**retry_index))

        assert last_record is not None
        records[condition.case] = last_record

    return records


def judge_pocqi_responses(
    *,
    question_id: str,
    question_text: str,
    specialty: str,
    responses: Sequence[PocqiResponseInput],
    judge_model: str,
    settings: PocqiJudgingSettings,
    model_caller: ModelCaller = call_model,
    output_paths: Mapping[PocqiJudgingCase, str | Path] | None = None,
    resume_tracker: PocqiResumeTracker | None = None,
    judging_cases: Sequence[PocqiJudgingCase] = tuple(PocqiJudgingCase),
    reveal_generator_identities: bool = False,
) -> dict[PocqiJudgingCase, PocqiJudgmentRecord]:
    """Run the three Real-POCQi judging conditions."""

    return judge_ranked_responses(
        question_id=question_id,
        question_text=question_text,
        specialty=specialty,
        responses=responses,
        judge_model=judge_model,
        settings=settings,
        profile=(
            IDENTITY_REVEALED_POCQI_JUDGING_PROFILE
            if reveal_generator_identities
            else POCQI_JUDGING_PROFILE
        ),
        model_caller=model_caller,
        output_paths=output_paths,
        resume_tracker=resume_tracker,
        judging_cases=judging_cases,
    )
