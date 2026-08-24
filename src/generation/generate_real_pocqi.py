"""Generate specialist answers for the Real-POCQi single-turn experiment.

The pipeline reads the committed Real-POCQi question set, reproduces the
experiment's deterministic sample, calls each requested model through the
unified inference layer, and appends one ``RealPocqiOutput`` record per API
attempt to a JSONL file. Successful logical generations are skipped on later
runs unless ``--force`` is supplied.

Run ``python -m generation.generate_real_pocqi --help`` for CLI options.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from inference import ModalConfig, ModelResponse, Provider, call_model, resolve_provider

from .real_pocqi import GenerationStatus, RealPocqiOutput


DEFAULT_INPUT = Path("data/question_sets/real_pocqi_questions.jsonl")
DEFAULT_OUTPUT = Path("data/outputs/generations/real_pocqi_generations.jsonl")
DEFAULT_EXPERIMENT_ID = "real_pocqi_single_turn_v1"
DEFAULT_PROMPT_TEMPLATE_ID = "specialist_answer_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPECIALIST_SYSTEM_PROMPT = """You are an experienced {specialty} specialist responding to a consult question from a generalist physician at the point of care.

Provide specific, evidence-based clinical guidance:
- Give a clear, actionable answer to the exact question asked.
- State concise clinical reasoning for your recommendation.
- Note key caveats, contraindications, monitoring, or when to escalate/refer.
- Be precise and specific; avoid generic boilerplate.
- Do not invent patient details, labs, or findings that were not provided."""

SPECIALIST_USER_PROMPT = """A generalist physician asks you, the {specialty} specialist, the following clinical question:

{question}

Provide your specialist response."""

ModelCaller = Callable[..., ModelResponse[Any]]
TRUNCATED_FINISH_REASONS = frozenset({"incomplete", "length", "max_tokens"})


@dataclass(frozen=True, slots=True)
class RealPocqiQuestion:
    """Question fields consumed by the generation pipeline."""

    question_id: str
    question_text: str
    specialty: str

    def __post_init__(self) -> None:
        for field_name in ("question_id", "question_text", "specialty"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Configuration shared by every generation in one invocation."""

    experiment_id: str
    run_id: str
    prompt_template_id: str
    temperature: float | None
    max_output_tokens: int
    retries: int
    retry_delay_seconds: float
    modal_config: ModalConfig | None = None


@dataclass(frozen=True, slots=True)
class ResumeState:
    """Existing terminal state recovered from the append-only output file."""

    succeeded_keys: frozenset[str]
    highest_attempt_by_key: dict[str, int]


class JsonlOutputStore:
    """Thread-safe, durable appends to the generation JSONL artifact."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: RealPocqiOutput) -> None:
        line = record.to_json() + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as output_file:
            output_file.write(line)
            output_file.flush()
            os.fsync(output_file.fileno())


def utc_now() -> str:
    """Return a UTC ISO-8601 timestamp suitable for serialized records."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def default_run_id() -> str:
    """Build a readable run ID with a suffix that prevents collisions."""

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"real-pocqi-{timestamp}-{uuid.uuid4().hex[:8]}"


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Load simple dotenv entries without replacing exported variables."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def infer_generator_family(model: str) -> str:
    """Return the model family independently of its inference transport."""

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
    if lowered.startswith("llama"):
        return "meta"
    return resolve_provider(model)[0].value


def load_questions(path: Path) -> list[RealPocqiQuestion]:
    """Load and validate Real-POCQi questions from the local JSONL artifact."""

    questions: list[RealPocqiQuestion] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                question = RealPocqiQuestion(
                    question_id=raw["question_id"],
                    question_text=raw["question_text"],
                    specialty=raw["specialty"],
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid question at {path}:{line_number}: {exc}") from exc
            if question.question_id in seen_ids:
                raise ValueError(f"duplicate question_id at {path}:{line_number}")
            seen_ids.add(question.question_id)
            questions.append(question)
    if not questions:
        raise ValueError(f"no questions found in {path}")
    return questions


def select_questions(
    questions: Sequence[RealPocqiQuestion],
    *,
    count: int | None,
    sample_seed: int,
    shuffle: bool,
    specialties: frozenset[str],
) -> list[RealPocqiQuestion]:
    """Filter and sample questions, matching ``datasets.Dataset.shuffle``."""

    selected = [
        question
        for question in questions
        if not specialties or question.specialty.casefold() in specialties
    ]
    if not selected:
        raise ValueError("the specialty filter selected no questions")

    if shuffle:
        # Hugging Face Dataset.shuffle is used by the original experiment. Using
        # it here preserves the exact seeded permutation instead of substituting
        # Python's different random.shuffle algorithm.
        from datasets import Dataset

        dataset = Dataset.from_list(
            [
                {
                    "question_id": question.question_id,
                    "question_text": question.question_text,
                    "specialty": question.specialty,
                }
                for question in selected
            ]
        ).shuffle(seed=sample_seed)
        selected = [RealPocqiQuestion(**row) for row in dataset]

    if count is not None:
        if count <= 0:
            raise ValueError("question count must be positive")
        if count > len(selected):
            raise ValueError(
                f"requested {count} questions, but only {len(selected)} are available"
            )
        selected = selected[:count]
    return selected


def load_resume_state(path: Path, experiment_id: str) -> ResumeState:
    """Recover successful keys and attempt counters from existing JSONL."""

    succeeded: set[str] = set()
    highest_attempt: dict[str, int] = {}
    if not path.exists():
        return ResumeState(frozenset(), {})

    with path.open(encoding="utf-8") as output_file:
        for line_number, line in enumerate(output_file, start=1):
            if not line.strip():
                continue
            try:
                record = RealPocqiOutput.from_json(line)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid output at {path}:{line_number}: {exc}") from exc
            if record.experiment_id != experiment_id:
                continue
            highest_attempt[record.generation_key] = max(
                record.attempt,
                highest_attempt.get(record.generation_key, 0),
            )
            if record.status is GenerationStatus.SUCCEEDED:
                succeeded.add(record.generation_key)
    return ResumeState(frozenset(succeeded), highest_attempt)


def generate_with_retries(
    question: RealPocqiQuestion,
    model: str,
    *,
    starting_attempt: int,
    settings: GenerationSettings,
    store: JsonlOutputStore,
    model_caller: ModelCaller = call_model,
) -> RealPocqiOutput:
    """Generate one logical answer, persisting every failed or successful attempt."""

    provider, _ = resolve_provider(model)
    system_prompt = SPECIALIST_SYSTEM_PROMPT.format(specialty=question.specialty)
    user_prompt = SPECIALIST_USER_PROMPT.format(
        specialty=question.specialty,
        question=question.question_text,
    )
    generation_key = RealPocqiOutput.build_generation_key(
        settings.experiment_id,
        question.question_id,
        model,
    )

    last_record: RealPocqiOutput | None = None
    for retry_index in range(settings.retries + 1):
        attempt = starting_attempt + retry_index
        started = time.perf_counter()
        common_fields: dict[str, Any] = {
            "experiment_id": settings.experiment_id,
            "run_id": settings.run_id,
            "generation_key": generation_key,
            "generation_id": f"gen-{uuid.uuid4()}",
            "attempt": attempt,
            "created_at": utc_now(),
            "question_id": question.question_id,
            "question_text": question.question_text,
            "specialty": question.specialty,
            "generator_family": infer_generator_family(model),
            "generator_model": model,
            "prompt_template_id": settings.prompt_template_id,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": settings.temperature,
            "max_output_tokens": settings.max_output_tokens,
            "seed": None,
        }
        try:
            request_options: dict[str, Any] = {}
            if settings.temperature is not None:
                request_options["temperature"] = settings.temperature
            response = model_caller(
                model,
                user_prompt,
                system=system_prompt,
                max_output_tokens=settings.max_output_tokens,
                modal_config=settings.modal_config if provider is Provider.MODAL else None,
                **request_options,
            )
            response_text = response.text.strip()
            if (response.finish_reason or "").casefold() in TRUNCATED_FINISH_REASONS:
                last_record = RealPocqiOutput(
                    **common_fields,
                    status=GenerationStatus.FAILED,
                    response_text=response_text or None,
                    generator_model_version=response.model,
                    finish_reason=response.finish_reason,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    latency_ms=round((time.perf_counter() - started) * 1000),
                    provider_request_id=response.request_id,
                    error_type="TruncatedResponseError",
                    error_message=(
                        "model response reached its output limit "
                        f"(finish_reason={response.finish_reason})"
                    ),
                )
                store.append(last_record)
                return last_record
            if not response_text:
                raise RuntimeError("model returned an empty response")
            last_record = RealPocqiOutput(
                **common_fields,
                status=GenerationStatus.SUCCEEDED,
                response_text=response_text,
                generator_model_version=response.model,
                finish_reason=response.finish_reason,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=round((time.perf_counter() - started) * 1000),
                provider_request_id=response.request_id,
            )
        except Exception as exc:  # Persist provider and network failures for auditability.
            last_record = RealPocqiOutput(
                **common_fields,
                status=GenerationStatus.FAILED,
                latency_ms=round((time.perf_counter() - started) * 1000),
                error_type=type(exc).__name__,
                error_message=str(exc) or repr(exc),
            )

        store.append(last_record)
        if last_record.status is GenerationStatus.SUCCEEDED:
            return last_record
        if retry_index < settings.retries and settings.retry_delay_seconds > 0:
            time.sleep(settings.retry_delay_seconds * (2**retry_index))

    assert last_record is not None
    return last_record


def write_manifest(
    path: Path,
    *,
    settings: GenerationSettings,
    input_path: Path,
    output_path: Path,
    models: Sequence[str],
    question_ids: Sequence[str],
    sample_seed: int,
    shuffle: bool,
    force: bool,
    planned: int,
    skipped: int,
    succeeded: int,
    failed: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write run-level configuration and completion counts next to the JSONL."""

    manifest = {
        "schema_version": "1.0",
        "experiment_id": settings.experiment_id,
        "run_id": settings.run_id,
        "created_at": utc_now(),
        "input": str(input_path),
        "output": str(output_path),
        "models": list(models),
        "question_ids": list(question_ids),
        "question_count": len(question_ids),
        "sample_seed": sample_seed,
        "shuffle": shuffle,
        "prompt_template_id": settings.prompt_template_id,
        "temperature": settings.temperature,
        "max_output_tokens": settings.max_output_tokens,
        "retries": settings.retries,
        "retry_delay_seconds": settings.retry_delay_seconds,
        "force": force,
        "logical_generations": {
            "planned": planned,
            "skipped_existing": skipped,
            "succeeded": succeeded,
            "failed": failed,
        },
    }
    if metadata:
        manifest.update(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Every model answers every selected question.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--prompt-template-id", default=DEFAULT_PROMPT_TEMPLATE_ID)
    parser.add_argument("--num-questions", type=int, default=125)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle with Hugging Face Dataset.shuffle before taking the sample.",
    )
    parser.add_argument(
        "--specialties",
        default=None,
        help="Optional comma-separated specialty names to include.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature; omitted by default for model compatibility.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Generate new attempts even when a successful logical generation exists.",
    )
    parser.add_argument("--modal-app", default=None)
    parser.add_argument("--modal-function", default="generate")
    parser.add_argument("--modal-environment", default=None)
    parser.add_argument(
        "--modal-transport",
        choices=("function", "openai_web"),
        default="function",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace, *, model_caller: ModelCaller = call_model) -> int:
    """Execute a configured generation run and return a process exit code."""

    if args.max_concurrency <= 0:
        raise ValueError("--max-concurrency must be positive")
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be positive")
    if args.retries < 0:
        raise ValueError("--retries cannot be negative")
    if args.retry_delay_seconds < 0:
        raise ValueError("--retry-delay-seconds cannot be negative")

    resolved_providers = [resolve_provider(model)[0] for model in args.models]
    modal_config = None
    if Provider.MODAL in resolved_providers:
        if not args.modal_app:
            raise ValueError("--modal-app is required when generating with a Modal model")
        modal_config = ModalConfig(
            app_name=args.modal_app,
            function_name=args.modal_function,
            environment_name=args.modal_environment,
            transport=args.modal_transport,
        )

    run_id = args.run_id or default_run_id()
    settings = GenerationSettings(
        experiment_id=args.experiment_id,
        run_id=run_id,
        prompt_template_id=args.prompt_template_id,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        retries=args.retries,
        retry_delay_seconds=args.retry_delay_seconds,
        modal_config=modal_config,
    )
    specialties = frozenset(
        item.strip().casefold()
        for item in (args.specialties or "").split(",")
        if item.strip()
    )
    questions = select_questions(
        load_questions(args.input),
        count=args.num_questions,
        sample_seed=args.sample_seed,
        shuffle=args.shuffle,
        specialties=specialties,
    )
    resume_state = load_resume_state(args.output, args.experiment_id)
    store = JsonlOutputStore(args.output)

    jobs: list[tuple[RealPocqiQuestion, str, str, int]] = []
    skipped = 0
    # Round-robin models within each question so mixed-provider runs share the
    # worker pool instead of exhausting one provider's entire queue first.
    for question in questions:
        for model in args.models:
            generation_key = RealPocqiOutput.build_generation_key(
                args.experiment_id,
                question.question_id,
                model,
            )
            if not args.force and generation_key in resume_state.succeeded_keys:
                skipped += 1
                continue
            starting_attempt = resume_state.highest_attempt_by_key.get(generation_key, 0) + 1
            jobs.append((question, model, generation_key, starting_attempt))

    print(
        f"Real-POCQi generation run {run_id}: {len(questions)} questions, "
        f"{len(args.models)} models, {len(jobs)} pending, {skipped} skipped"
    )
    succeeded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        futures = {
            executor.submit(
                generate_with_retries,
                question,
                model,
                starting_attempt=starting_attempt,
                settings=settings,
                store=store,
                model_caller=model_caller,
            ): (question.question_id, model)
            for question, model, _generation_key, starting_attempt in jobs
        }
        for future in as_completed(futures):
            question_id, model = futures[future]
            record = future.result()
            if record.status is GenerationStatus.SUCCEEDED:
                succeeded += 1
            else:
                failed += 1
                print(
                    f"FAILED {question_id} with {model}: "
                    f"{record.error_type}: {record.error_message}"
                )

    manifest_path = args.output.parent / f"{run_id}.manifest.json"
    write_manifest(
        manifest_path,
        settings=settings,
        input_path=args.input,
        output_path=args.output,
        models=args.models,
        question_ids=[question.question_id for question in questions],
        sample_seed=args.sample_seed,
        shuffle=args.shuffle,
        force=args.force,
        planned=len(jobs),
        skipped=skipped,
        succeeded=succeeded,
        failed=failed,
    )
    print(
        f"Complete: {succeeded} succeeded, {failed} failed; "
        f"outputs={args.output}; manifest={manifest_path}"
    )
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
