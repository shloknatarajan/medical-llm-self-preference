"""Generate the Real-POCQi Qwen cohort on Modal with batched vLLM inference.

Usage:
    uv run modal run src/generation/modal_real_pocqi_generation.py

The 27B checkpoint runs on one H100; the 122B-A10B checkpoint uses tensor
parallel inference on two H100s. Both use official FP8 checkpoints and have
thinking disabled to match the experiment's direct-response condition.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import modal

from generation.generate_real_pocqi import (
    DEFAULT_EXPERIMENT_ID,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    DEFAULT_PROMPT_TEMPLATE_ID,
    SPECIALIST_SYSTEM_PROMPT,
    SPECIALIST_USER_PROMPT,
    GenerationSettings,
    JsonlOutputStore,
    RealPocqiQuestion,
    default_run_id,
    load_questions,
    load_resume_state,
    select_questions,
    utc_now,
    write_manifest,
)
from generation.real_pocqi import GenerationStatus, RealPocqiOutput


APP_NAME = "medical-real-pocqi-qwen-generation"
QWEN_27B = "Qwen/Qwen3.8-27B-FP8"
QWEN_122B = "Qwen/Qwen3.5-122B-A10B-FP8"
MODEL_IDS = (QWEN_27B, QWEN_122B)
TRUNCATED_FINISH_REASONS = frozenset({"incomplete", "length", "max_tokens"})
MINUTES = 60

app = modal.App(APP_NAME)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.27.1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_USE_DEEP_GEMM": "0"})
    .add_local_python_source("generation", "inference")
)
hf_cache = modal.Volume.from_name("medical-llm-huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("medical-llm-vllm-cache", create_if_missing=True)
volumes = {
    "/root/.cache/huggingface": hf_cache,
    "/root/.cache/vllm": vllm_cache,
}


def _generate_batch(
    llm: Any,
    conversations: list[list[dict[str, str]]],
    *,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    import vllm

    sampling = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    started = time.perf_counter()
    outputs = llm.chat(
        conversations,
        sampling_params=sampling,
        chat_template_kwargs={"enable_thinking": False},
    )
    return {
        "texts": [output.outputs[0].text.strip() for output in outputs],
        "input_tokens": [len(output.prompt_token_ids) for output in outputs],
        "output_tokens": [len(output.outputs[0].token_ids) for output in outputs],
        "finish_reasons": [output.outputs[0].finish_reason for output in outputs],
        "batch_latency_seconds": time.perf_counter() - started,
    }


@app.cls(
    image=vllm_image,
    gpu="H100",
    timeout=MINUTES * 60,
    startup_timeout=MINUTES * 60,
    scaledown_window=5 * MINUTES,
    volumes=volumes,
)
class Qwen27B:
    @modal.enter()
    def start(self) -> None:
        import vllm

        self.llm = vllm.LLM(
            model=QWEN_27B,
            max_model_len=8192,
            gpu_memory_utilization=0.92,
            limit_mm_per_prompt={"image": 0, "video": 0},
            enforce_eager=True,
        )

    @modal.method()
    def generate_batch(
        self,
        conversations: list[list[dict[str, str]]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        return _generate_batch(
            self.llm,
            conversations,
            temperature=temperature,
            max_tokens=max_tokens,
        )


@app.cls(
    image=vllm_image,
    gpu="H100:2",
    timeout=MINUTES * 60,
    startup_timeout=MINUTES * 60,
    scaledown_window=5 * MINUTES,
    volumes=volumes,
)
class Qwen122B:
    @modal.enter()
    def start(self) -> None:
        import vllm

        self.llm = vllm.LLM(
            model=QWEN_122B,
            tensor_parallel_size=2,
            max_model_len=8192,
            gpu_memory_utilization=0.92,
            limit_mm_per_prompt={"image": 0, "video": 0},
            enforce_eager=True,
        )

    @modal.method()
    def generate_batch(
        self,
        conversations: list[list[dict[str, str]]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        return _generate_batch(
            self.llm,
            conversations,
            temperature=temperature,
            max_tokens=max_tokens,
        )


@dataclass(frozen=True, slots=True)
class PendingGeneration:
    question: RealPocqiQuestion
    starting_attempt: int


def _conversation(question: RealPocqiQuestion) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SPECIALIST_SYSTEM_PROMPT.format(specialty=question.specialty),
        },
        {
            "role": "user",
            "content": SPECIALIST_USER_PROMPT.format(
                specialty=question.specialty,
                question=question.question_text,
            ),
        },
    ]


def _record(
    *,
    pending: PendingGeneration,
    model: str,
    settings: GenerationSettings,
    attempt: int,
    status: GenerationStatus,
    latency_ms: int,
    response_text: str | None = None,
    finish_reason: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    error: Exception | None = None,
) -> RealPocqiOutput:
    question = pending.question
    return RealPocqiOutput(
        experiment_id=settings.experiment_id,
        run_id=settings.run_id,
        generation_key=RealPocqiOutput.build_generation_key(
            settings.experiment_id,
            question.question_id,
            model,
        ),
        generation_id=f"gen-{uuid.uuid4()}",
        attempt=attempt,
        status=status,
        created_at=utc_now(),
        question_id=question.question_id,
        question_text=question.question_text,
        specialty=question.specialty,
        generator_family="qwen",
        generator_model=model,
        generator_model_version=model if status is GenerationStatus.SUCCEEDED else None,
        prompt_template_id=settings.prompt_template_id,
        system_prompt=SPECIALIST_SYSTEM_PROMPT.format(specialty=question.specialty),
        user_prompt=SPECIALIST_USER_PROMPT.format(
            specialty=question.specialty,
            question=question.question_text,
        ),
        response_text=response_text,
        finish_reason=finish_reason,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        seed=None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        error_type=type(error).__name__ if error else None,
        error_message=(str(error) or repr(error)) if error else None,
    )


def _run_batch(
    *,
    remote_model: Any,
    model: str,
    pending: Sequence[PendingGeneration],
    settings: GenerationSettings,
    store: JsonlOutputStore,
) -> tuple[int, int]:
    remaining = list(pending)
    succeeded = 0

    for retry_index in range(settings.retries + 1):
        if not remaining:
            break
        conversations = [_conversation(item.question) for item in remaining]
        started = time.perf_counter()
        try:
            result = remote_model.generate_batch.remote(
                conversations,
                settings.temperature,
                settings.max_output_tokens,
            )
            expected = len(remaining)
            fields = ("texts", "input_tokens", "output_tokens", "finish_reasons")
            if any(len(result[field]) != expected for field in fields):
                raise RuntimeError("Modal generation returned an unexpected batch size")
            per_item_latency_ms = round(
                float(result["batch_latency_seconds"]) * 1000 / expected
            )
            retry_items: list[PendingGeneration] = []
            for index, item in enumerate(remaining):
                attempt = item.starting_attempt + retry_index
                text = str(result["texts"][index]).strip()
                if text:
                    store.append(
                        _record(
                            pending=item,
                            model=model,
                            settings=settings,
                            attempt=attempt,
                            status=GenerationStatus.SUCCEEDED,
                            latency_ms=per_item_latency_ms,
                            response_text=text,
                            finish_reason=result["finish_reasons"][index],
                            input_tokens=int(result["input_tokens"][index]),
                            output_tokens=int(result["output_tokens"][index]),
                        )
                    )
                    succeeded += 1
                else:
                    error = RuntimeError("model returned an empty response")
                    store.append(
                        _record(
                            pending=item,
                            model=model,
                            settings=settings,
                            attempt=attempt,
                            status=GenerationStatus.FAILED,
                            latency_ms=per_item_latency_ms,
                            error=error,
                        )
                    )
                    retry_items.append(item)
            remaining = retry_items
        except Exception as exc:
            latency_ms = round((time.perf_counter() - started) * 1000 / len(remaining))
            for item in remaining:
                store.append(
                    _record(
                        pending=item,
                        model=model,
                        settings=settings,
                        attempt=item.starting_attempt + retry_index,
                        status=GenerationStatus.FAILED,
                        latency_ms=latency_ms,
                        error=exc,
                    )
                )

        if remaining and retry_index < settings.retries:
            time.sleep(settings.retry_delay_seconds * (2**retry_index))

    return succeeded, len(remaining)


def _chunks(items: Sequence[PendingGeneration], size: int) -> list[list[PendingGeneration]]:
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def _truncated_keys_below_cap(
    path: Path,
    experiment_id: str,
    max_output_tokens: int,
) -> frozenset[str]:
    """Find latest successful attempts truncated under a smaller token cap."""

    latest: dict[str, RealPocqiOutput] = {}
    if not path.exists():
        return frozenset()
    with path.open(encoding="utf-8") as output_file:
        for line_number, line in enumerate(output_file, start=1):
            if not line.strip():
                continue
            try:
                record = RealPocqiOutput.from_json(line)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid output at {path}:{line_number}: {exc}") from exc
            if (
                record.experiment_id == experiment_id
                and record.status is GenerationStatus.SUCCEEDED
                and record.attempt >= latest.get(record.generation_key, record).attempt
            ):
                latest[record.generation_key] = record
    return frozenset(
        key
        for key, record in latest.items()
        if record.finish_reason in TRUNCATED_FINISH_REASONS
        and (record.max_output_tokens or 0) < max_output_tokens
    )


def _run_model(
    *,
    remote_model: Any,
    model: str,
    questions: Sequence[RealPocqiQuestion],
    settings: GenerationSettings,
    store: JsonlOutputStore,
    output_path: Path,
    batch_size: int,
    force: bool,
    regenerate_truncated: bool,
) -> tuple[int, int, int, int]:
    resume = load_resume_state(output_path, settings.experiment_id)
    truncated_keys = (
        _truncated_keys_below_cap(
            output_path,
            settings.experiment_id,
            settings.max_output_tokens,
        )
        if regenerate_truncated
        else frozenset()
    )
    pending: list[PendingGeneration] = []
    skipped = 0
    for question in questions:
        key = RealPocqiOutput.build_generation_key(
            settings.experiment_id,
            question.question_id,
            model,
        )
        if not force and key in resume.succeeded_keys and key not in truncated_keys:
            skipped += 1
            continue
        pending.append(
            PendingGeneration(
                question=question,
                starting_attempt=resume.highest_attempt_by_key.get(key, 0) + 1,
            )
        )

    print(f"{model}: {len(pending)} pending, {skipped} already complete")
    succeeded = 0
    failed = 0
    batches = _chunks(pending, batch_size)
    for batch_number, batch in enumerate(batches, start=1):
        batch_succeeded, batch_failed = _run_batch(
            remote_model=remote_model,
            model=model,
            pending=batch,
            settings=settings,
            store=store,
        )
        succeeded += batch_succeeded
        failed += batch_failed
        print(
            f"{model}: batch {batch_number}/{len(batches)} complete "
            f"({batch_succeeded} succeeded, {batch_failed} failed)"
        )
    return len(pending), skipped, succeeded, failed


@app.local_entrypoint()
def main(
    input_path: str = str(DEFAULT_INPUT),
    output_path: str = str(DEFAULT_OUTPUT),
    num_questions: int = 620,
    sample_seed: int = 42,
    temperature: float = 0.3,
    max_output_tokens: int = 2048,
    batch_size: int = 8,
    retries: int = 2,
    retry_delay_seconds: float = 2.0,
    run_id: str = "",
    models: str = ",".join(MODEL_IDS),
    force: bool = False,
    regenerate_truncated: bool = False,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    selected_models = tuple(item.strip() for item in models.split(",") if item.strip())
    unknown_models = set(selected_models) - set(MODEL_IDS)
    if unknown_models:
        raise ValueError(f"unsupported models: {sorted(unknown_models)}")

    selected_questions = select_questions(
        load_questions(Path(input_path)),
        count=num_questions,
        sample_seed=sample_seed,
        shuffle=True,
        specialties=frozenset(),
    )
    destination = Path(output_path)
    store = JsonlOutputStore(destination)
    settings = GenerationSettings(
        experiment_id=DEFAULT_EXPERIMENT_ID,
        run_id=run_id or default_run_id(),
        prompt_template_id=DEFAULT_PROMPT_TEMPLATE_ID,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
    )

    remotes = {
        QWEN_27B: Qwen27B(),
        QWEN_122B: Qwen122B(),
    }
    planned = skipped = succeeded = failed = 0
    for model in selected_models:
        model_planned, model_skipped, model_succeeded, model_failed = _run_model(
            remote_model=remotes[model],
            model=model,
            questions=selected_questions,
            settings=settings,
            store=store,
            output_path=destination,
            batch_size=batch_size,
            force=force,
            regenerate_truncated=regenerate_truncated,
        )
        planned += model_planned
        skipped += model_skipped
        succeeded += model_succeeded
        failed += model_failed

    manifest_path = destination.parent / f"{settings.run_id}.manifest.json"
    write_manifest(
        manifest_path,
        settings=settings,
        input_path=Path(input_path),
        output_path=destination,
        models=selected_models,
        question_ids=[question.question_id for question in selected_questions],
        sample_seed=sample_seed,
        shuffle=True,
        force=force,
        planned=planned,
        skipped=skipped,
        succeeded=succeeded,
        failed=failed,
        metadata={"regenerate_truncated": regenerate_truncated},
    )
    print(
        f"Generation complete: {succeeded} succeeded, {failed} failed, "
        f"{skipped} skipped; output={destination}; manifest={manifest_path}"
    )
    if failed:
        raise RuntimeError(f"{failed} logical generations failed after retries")
