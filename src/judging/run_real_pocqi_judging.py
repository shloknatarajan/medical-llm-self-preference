"""Run all three Real-POCQi judging styles across questions and API judges.

The runner consumes the latest successful generation for every requested
question/model cell, validates complete candidate coverage, and schedules one
three-call judging job per question and judge model.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from inference import ModelResponse, Provider, call_model, resolve_provider

from generation.generate_real_pocqi import load_dotenv
from generation.real_pocqi import GenerationStatus, RealPocqiOutput

from .judge_real_pocqi import (
    PocqiJudgingSettings,
    PocqiResumeTracker,
    PocqiResponseInput,
    build_pocqi_judgment_keys,
    judge_pocqi_responses,
)
from .real_pocqi import PocqiJudgingCase, PocqiJudgmentStatus


DEFAULT_GENERATIONS_PATH = Path(
    "data/outputs/generations/real_pocqi_generations.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("data/real_pcoqi/judgements")
DEFAULT_EXPERIMENT_ID = "real_pocqi_all_judges_v1"

DEFAULT_GENERATOR_MODELS = (
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.7-flash",
    "Qwen/Qwen3.5-122B-A10B-FP8",
    "Qwen/Qwen3.8-27B-FP8",
)
DEFAULT_JUDGE_MODELS = (
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.7-flash",
)

ModelCaller = Callable[..., ModelResponse[Any]]


@dataclass(frozen=True, slots=True)
class PocqiQuestionResponses:
    """Complete candidate set for one Real-POCQi question."""

    question_id: str
    question_text: str
    specialty: str
    responses: tuple[PocqiResponseInput, ...]


@dataclass(frozen=True, slots=True)
class PlannedJob:
    """One question judged by one model across the three conditions."""

    question: PocqiQuestionResponses
    judge_model: str
    pending_cases: tuple[PocqiJudgingCase, ...]


class ProviderLimitedCaller:
    """Apply independent concurrency limits around each provider call."""

    def __init__(
        self,
        caller: ModelCaller,
        limits: dict[Provider, int],
    ) -> None:
        self._caller = caller
        self._semaphores = {
            provider: threading.BoundedSemaphore(limit)
            for provider, limit in limits.items()
        }

    def __call__(self, model: str, input: str, **kwargs: Any) -> ModelResponse[Any]:
        provider, _ = resolve_provider(model)
        with self._semaphores[provider]:
            return self._caller(model, input, **kwargs)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def default_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"real-pocqi-judging-{timestamp}-{uuid.uuid4().hex[:8]}"


def output_paths(output_dir: Path) -> dict[PocqiJudgingCase, Path]:
    return {
        PocqiJudgingCase.DIRECT_RANKING: output_dir / "direct_ranking.jsonl",
        PocqiJudgingCase.RUBRIC_SUM_RANKING: (
            output_dir / "rubric_sum_ranking.jsonl"
        ),
        PocqiJudgingCase.RUBRIC_AND_MODEL_RANKING: (
            output_dir / "rubric_and_model_ranking.jsonl"
        ),
    }


def load_latest_question_responses(
    path: Path,
    *,
    generator_models: Sequence[str],
    num_questions: int | None = None,
) -> list[PocqiQuestionResponses]:
    """Load latest successful generations and require a complete model matrix."""

    requested_models = tuple(generator_models)
    if len(set(requested_models)) != len(requested_models):
        raise ValueError("generator models must be unique")
    requested_set = set(requested_models)
    latest: dict[tuple[str, str], RealPocqiOutput] = {}
    question_order: list[str] = []
    seen_questions: set[str] = set()

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = RealPocqiOutput.from_json(line)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid generation at {path}:{line_number}: {exc}"
                ) from exc
            if (
                record.status is not GenerationStatus.SUCCEEDED
                or record.generator_model not in requested_set
            ):
                continue
            if record.question_id not in seen_questions:
                question_order.append(record.question_id)
                seen_questions.add(record.question_id)
            key = (record.question_id, record.generator_model)
            previous = latest.get(key)
            if previous is None or record.attempt >= previous.attempt:
                latest[key] = record

    if not latest:
        raise ValueError(f"no requested successful generations found in {path}")
    if num_questions is not None:
        if num_questions <= 0:
            raise ValueError("num_questions must be positive")
        question_order = question_order[:num_questions]

    incomplete: dict[str, list[str]] = {}
    questions: list[PocqiQuestionResponses] = []
    for question_id in question_order:
        missing = [
            model
            for model in requested_models
            if (question_id, model) not in latest
        ]
        if missing:
            incomplete[question_id] = missing
            continue
        records = [latest[(question_id, model)] for model in requested_models]
        question_texts = {record.question_text for record in records}
        specialties = {record.specialty for record in records}
        if len(question_texts) != 1 or len(specialties) != 1:
            raise ValueError(f"inconsistent question metadata for {question_id}")
        questions.append(
            PocqiQuestionResponses(
                question_id=question_id,
                question_text=records[0].question_text,
                specialty=records[0].specialty,
                responses=tuple(
                    PocqiResponseInput(
                        generation_id=record.generation_id,
                        generator_family=record.generator_family,
                        generator_model=record.generator_model,
                        response_text=record.response_text or "",
                    )
                    for record in records
                ),
            )
        )
    if incomplete:
        sample = dict(list(incomplete.items())[:5])
        raise ValueError(
            f"{len(incomplete)} questions have incomplete generation coverage; "
            f"first missing cells: {sample}"
        )
    if not questions:
        raise ValueError("no complete questions available for judging")
    return questions


def validate_judge_models(judge_models: Sequence[str]) -> tuple[str, ...]:
    models = tuple(judge_models)
    if not models:
        raise ValueError("at least one judge model is required")
    if len(set(models)) != len(models):
        raise ValueError("judge models must be unique")
    for model in models:
        provider, _ = resolve_provider(model)
        if provider not in (Provider.OPENAI, Provider.ANTHROPIC, Provider.GEMINI):
            raise ValueError(
                f"judge {model!r} uses {provider.value}; this runner currently "
                "supports only OpenAI, Anthropic, and Gemini"
            )
    return models


def plan_jobs(
    *,
    questions: Sequence[PocqiQuestionResponses],
    judge_models: Sequence[str],
    settings: PocqiJudgingSettings,
    paths: dict[PocqiJudgingCase, Path],
    tracker: PocqiResumeTracker,
    judging_cases: Sequence[PocqiJudgingCase],
) -> list[PlannedJob]:
    jobs: list[PlannedJob] = []
    for question in questions:
        for judge_model in judge_models:
            keys = build_pocqi_judgment_keys(
                question_id=question.question_id,
                responses=question.responses,
                judge_model=judge_model,
                settings=settings,
                judging_cases=judging_cases,
            )
            pending: list[PocqiJudgingCase] = []
            for case, key in keys.items():
                succeeded, _ = tracker.lookup(paths[case], key)
                if settings.force or succeeded is None:
                    pending.append(case)
            jobs.append(
                PlannedJob(
                    question=question,
                    judge_model=judge_model,
                    pending_cases=tuple(pending),
                )
            )
    return jobs


def write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    run_id: str,
    questions: Sequence[PocqiQuestionResponses],
    judge_models: Sequence[str],
    counts: dict[str, int],
) -> None:
    manifest = {
        "schema_version": "1.0",
        "experiment_id": args.experiment_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "input_generations": str(args.input_generations),
        "output_dir": str(args.output_dir),
        "question_count": len(questions),
        "question_ids": [question.question_id for question in questions],
        "generator_models": list(args.generator_models),
        "judge_models": list(judge_models),
        "judging_cases": list(args.judging_cases),
        "presentation_seed": args.presentation_seed,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "retries": args.retries,
        "retry_delay_seconds": args.retry_delay_seconds,
        "force": args.force,
        "provider_concurrency": {
            "openai": args.openai_concurrency,
            "anthropic": args.anthropic_concurrency,
            "gemini": args.gemini_concurrency,
        },
        "logical_judgments": counts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-generations",
        type=Path,
        default=DEFAULT_GENERATIONS_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generator-models", nargs="+", default=DEFAULT_GENERATOR_MODELS)
    parser.add_argument("--judge-models", nargs="+", default=DEFAULT_JUDGE_MODELS)
    parser.add_argument(
        "--judging-cases",
        nargs="+",
        choices=[case.value for case in PocqiJudgingCase],
        default=[case.value for case in PocqiJudgingCase],
    )
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--presentation-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-concurrency", type=int, default=6)
    parser.add_argument("--openai-concurrency", type=int, default=2)
    parser.add_argument("--anthropic-concurrency", type=int, default=2)
    parser.add_argument("--gemini-concurrency", type=int, default=2)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def run(
    args: argparse.Namespace,
    *,
    model_caller: ModelCaller = call_model,
) -> int:
    """Execute or dry-run the batch judging matrix."""

    for name in (
        "max_concurrency",
        "openai_concurrency",
        "anthropic_concurrency",
        "gemini_concurrency",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.retries < 0:
        raise ValueError("--retries cannot be negative")
    if args.retry_delay_seconds < 0:
        raise ValueError("--retry-delay-seconds cannot be negative")

    judge_models = validate_judge_models(args.judge_models)
    judging_cases = tuple(PocqiJudgingCase(value) for value in args.judging_cases)
    questions = load_latest_question_responses(
        args.input_generations,
        generator_models=args.generator_models,
        num_questions=args.num_questions,
    )
    run_id = args.run_id or default_run_id()
    settings = PocqiJudgingSettings(
        experiment_id=args.experiment_id,
        run_id=run_id,
        presentation_seed=args.presentation_seed,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        retries=args.retries,
        retry_delay_seconds=args.retry_delay_seconds,
        force=args.force,
    )
    paths = output_paths(args.output_dir)
    tracker = PocqiResumeTracker(list(paths.values()))
    jobs = plan_jobs(
        questions=questions,
        judge_models=judge_models,
        settings=settings,
        paths=paths,
        tracker=tracker,
        judging_cases=judging_cases,
    )
    total_logical = len(jobs) * len(judging_cases)
    pending_logical = sum(len(job.pending_cases) for job in jobs)
    skipped = total_logical - pending_logical
    pending_jobs = [job for job in jobs if job.pending_cases]
    print(
        f"Real-POCQi judging {run_id}: {len(questions)} questions, "
        f"{len(judge_models)} judges, {total_logical} logical judgments, "
        f"{pending_logical} pending, {skipped} skipped"
    )
    if args.dry_run:
        return 0

    limited_caller = ProviderLimitedCaller(
        model_caller,
        {
            Provider.OPENAI: args.openai_concurrency,
            Provider.ANTHROPIC: args.anthropic_concurrency,
            Provider.GEMINI: args.gemini_concurrency,
        },
    )

    succeeded = failed = completed_jobs = 0

    def execute(job: PlannedJob):
        question = job.question
        return job, judge_pocqi_responses(
            question_id=question.question_id,
            question_text=question.question_text,
            specialty=question.specialty,
            responses=question.responses,
            judge_model=job.judge_model,
            settings=settings,
            model_caller=limited_caller,
            output_paths=paths,
            resume_tracker=tracker,
            judging_cases=job.pending_cases,
        )

    with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        futures = {executor.submit(execute, job): job for job in pending_jobs}
        for future in as_completed(futures):
            job, records = future.result()
            for case in job.pending_cases:
                if records[case].status is PocqiJudgmentStatus.SUCCEEDED:
                    succeeded += 1
                else:
                    failed += 1
                    record = records[case]
                    print(
                        f"FAILED {job.question.question_id} {job.judge_model} "
                        f"{case.value}: {record.error_type}: {record.error_message}"
                    )
            completed_jobs += 1
            if completed_jobs % 25 == 0 or completed_jobs == len(pending_jobs):
                print(
                    f"Progress: {completed_jobs}/{len(pending_jobs)} jobs; "
                    f"{succeeded} judgments succeeded, {failed} failed"
                )

    counts = {
        "total": total_logical,
        "pending_at_start": pending_logical,
        "skipped_existing": skipped,
        "succeeded": succeeded,
        "failed": failed,
    }
    manifest_path = args.output_dir / f"{run_id}.manifest.json"
    write_manifest(
        manifest_path,
        args=args,
        run_id=run_id,
        questions=questions,
        judge_models=judge_models,
        counts=counts,
    )
    print(
        f"Complete: {succeeded} succeeded, {failed} failed, {skipped} skipped; "
        f"manifest={manifest_path}"
    )
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
