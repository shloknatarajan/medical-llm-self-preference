"""Generate role-separated MedSP1000 conversations on Modal without judging."""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import modal

from generation.medsp1000 import (
    CLINICIAN_PROMPT_TEMPLATE_ID,
    CLINICIAN_SYSTEM_PROMPT,
    CLINICIAN_TURN_PROMPT_TEMPLATE,
    DEFAULT_CLINICIAN_MAX_TOKENS,
    DEFAULT_CLINICIAN_MODEL,
    DEFAULT_EXCHANGES,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    EXPERIMENT_ID,
    MedSPQuestion,
    OUTPUT_SCHEMA_VERSION,
    PATIENT_MODEL,
    PATIENT_PROMPT_TEMPLATE_ID,
    PATIENT_SYSTEM_PROMPT,
    PATIENT_TURN_GROUNDING_REMINDER,
    PROMPT_VERSION,
    advance_clinician_history,
    advance_patient_history,
    append_jsonl,
    build_generation_key,
    chunked,
    clinician_generation_messages,
    clinician_messages,
    default_run_id,
    generation_record,
    load_questions,
    load_resume_state,
    patient_generation_messages,
    patient_messages,
    project_root,
    utc_now,
)
from inference import ModelResponse, Provider, call_model, resolve_provider


APP_NAME = "medical-medsp1000-generation"
MINUTES = 60
SEED = 20260824
PROJECT_ROOT = project_root(Path(__file__))
QWEN_27B_CLINICIAN_MODEL = "Qwen/Qwen3.8-27B-FP8"
TRUNCATED_FINISH_REASONS = frozenset(
    {"incomplete", "length", "max_tokens", "max_output_tokens"}
)

app = modal.App(APP_NAME)
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.27.1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_USE_DEEP_GEMM": "0"})
    .add_local_python_source("generation")
    .add_local_python_source("inference")
)
hf_cache = modal.Volume.from_name("medical-llm-huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("medical-llm-vllm-cache", create_if_missing=True)
volumes = {
    "/root/.cache/huggingface": hf_cache,
    "/root/.cache/vllm": vllm_cache,
}


def _generate(
    llm: Any,
    conversations: list[list[dict[str, str]]],
    *,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    import vllm

    sampling = vllm.SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_tokens,
        seed=SEED,
    )
    started = time.perf_counter()
    outputs = llm.chat(
        conversations,
        sampling_params=sampling,
        chat_template_kwargs={"enable_thinking": False},
    )
    batch_latency_seconds = time.perf_counter() - started
    latency_ms = round(batch_latency_seconds * 1000 / len(outputs))
    return {
        "texts": [output.outputs[0].text.strip() for output in outputs],
        "input_tokens": [len(output.prompt_token_ids) for output in outputs],
        "output_tokens": [len(output.outputs[0].token_ids) for output in outputs],
        "finish_reasons": [output.outputs[0].finish_reason for output in outputs],
        "latency_ms": [latency_ms] * len(outputs),
    }


def _validate_batch(
    result: dict[str, Any], expected_count: int, role: str, exchange: int
) -> list[str]:
    texts = result["texts"]
    finish_reasons = result["finish_reasons"]
    metadata_fields = (
        "input_tokens",
        "output_tokens",
        "finish_reasons",
        "latency_ms",
    )
    if any(
        not isinstance(result.get(field), list)
        or len(result[field]) != expected_count
        for field in metadata_fields
    ):
        raise RuntimeError(
            f"invalid {role} metadata batch at exchange {exchange}"
        )
    if len(texts) != expected_count or any(not text for text in texts):
        raise RuntimeError(f"invalid {role} batch at exchange {exchange}")
    if len(finish_reasons) != expected_count or any(
        not reason for reason in finish_reasons
    ):
        raise RuntimeError(
            f"missing {role} finish reason at exchange {exchange}: {finish_reasons}"
        )
    if any(
        str(reason).casefold() in TRUNCATED_FINISH_REASONS
        for reason in finish_reasons
    ):
        raise RuntimeError(
            f"truncated {role} batch at exchange {exchange}: {finish_reasons}"
        )
    return texts


def _call_api_clinician(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> tuple[ModelResponse[Any], int]:
    started = time.perf_counter()
    response = call_model(model, messages, max_output_tokens=max_tokens)
    return response, round((time.perf_counter() - started) * 1000)


def _generate_api_clinician_batch(
    model: str,
    conversations: list[list[dict[str, str]]],
    max_tokens: int,
) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=len(conversations)) as executor:
        results = list(
            executor.map(
                lambda messages: _call_api_clinician(model, messages, max_tokens),
                conversations,
            )
        )
    responses = [result[0] for result in results]
    return {
        "texts": [response.text.strip() for response in responses],
        "input_tokens": [response.usage.input_tokens or 0 for response in responses],
        "output_tokens": [response.usage.output_tokens or 0 for response in responses],
        "finish_reasons": [response.finish_reason or "unknown" for response in responses],
        "model_versions": [response.model for response in responses],
        "latency_ms": [result[1] for result in results],
    }


def _load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_manifest(
    path: Path,
    *,
    run_id: str,
    input_path: Path,
    output_path: Path,
    question_ids: list[str],
    exchanges: int,
    patient_max_tokens: int,
    clinician_max_tokens: int,
    clinician_model: str,
    clinician_temperature: float | None,
    force: bool,
    batch_size: int,
    planned: int,
    skipped: int,
    succeeded: int,
    failed: int,
    checkpoint_count: int,
    run_status: str,
    created_at: str,
    error: BaseException | None = None,
) -> None:
    if run_status not in {"running", "completed", "interrupted"}:
        raise ValueError(f"unsupported run status: {run_status}")
    checkpointed_at = utc_now()
    manifest = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "created_at": created_at,
        "updated_at": checkpointed_at,
        "run_status": run_status,
        "input": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "output": str(output_path),
        "question_ids": question_ids,
        "question_count": len(question_ids),
        "patient_model": PATIENT_MODEL,
        "clinician_model": clinician_model,
        "clinician_provider": resolve_provider(clinician_model)[0].value,
        "prompt_version": PROMPT_VERSION,
        "prompt_templates": {
            "patient": {
                "template_id": PATIENT_PROMPT_TEMPLATE_ID,
                "system_sha256": _sha256_text(PATIENT_SYSTEM_PROMPT),
                "system_text": PATIENT_SYSTEM_PROMPT,
                "turn_reminder_sha256": _sha256_text(
                    PATIENT_TURN_GROUNDING_REMINDER
                ),
                "turn_reminder_text": PATIENT_TURN_GROUNDING_REMINDER,
            },
            "clinician": {
                "template_id": CLINICIAN_PROMPT_TEMPLATE_ID,
                "system_sha256": _sha256_text(CLINICIAN_SYSTEM_PROMPT),
                "system_text": CLINICIAN_SYSTEM_PROMPT,
                "turn_template_sha256": _sha256_text(
                    CLINICIAN_TURN_PROMPT_TEMPLATE
                ),
                "turn_template_text": CLINICIAN_TURN_PROMPT_TEMPLATE,
            },
        },
        "exchange_count": exchanges,
        "patient_temperature": 0.2,
        "clinician_temperature": clinician_temperature,
        "patient_max_output_tokens": patient_max_tokens,
        "clinician_max_output_tokens": clinician_max_tokens,
        "seed": SEED,
        "reasoning_enabled": False,
        "patient_inference_engine": "vllm==0.27.1 on Modal",
        "clinician_inference_engine": (
            "vllm==0.27.1 on Modal"
            if resolve_provider(clinician_model)[0] is Provider.MODAL
            else "provider API"
        ),
        "patient_max_model_len": 16384,
        "clinician_max_model_len": (
            8192
            if resolve_provider(clinician_model)[0] is Provider.MODAL
            else None
        ),
        "force": force,
        "checkpointing": {
            "unit": "complete_conversation",
            "batch_size": batch_size,
            "checkpoint_count": checkpoint_count,
            "last_checkpoint_at": checkpointed_at,
        },
        "logical_generations": {
            "planned": planned,
            "skipped_existing": skipped,
            "succeeded": succeeded,
            "failed": failed,
            "remaining": planned - succeeded - failed,
        },
        "error_type": type(error).__name__ if error else None,
        "error_message": (str(error) or repr(error)) if error else None,
        "generation_only": True,
        "grading_or_judging_performed": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as manifest_file:
        manifest_file.write(json.dumps(manifest, indent=2) + "\n")
        manifest_file.flush()
        os.fsync(manifest_file.fileno())
    temporary_path.replace(path)


def _generate_conversation_batch(
    *,
    questions: list[MedSPQuestion],
    clinician_remote: Any | None,
    clinician_model: str,
    clinician_temperature: float | None,
    patient: Any,
    run_id: str,
    starting_attempts: dict[str, int],
    exchanges: int,
    patient_max_tokens: int,
    clinician_max_tokens: int,
    batch_number: int,
    batch_count: int,
) -> list[dict[str, Any]]:
    clinician_histories = [
        clinician_messages(question, exchanges) for question in questions
    ]
    patient_histories = [patient_messages(question) for question in questions]
    turns: list[list[dict[str, Any]]] = [[] for _ in questions]
    prior_patient_replies: list[str | None] = [None] * len(questions)

    try:
        for exchange in range(1, exchanges + 1):
            clinician_prompts = [
                clinician_generation_messages(
                    history, prior_patient_replies[index], exchange, exchanges
                )
                for index, history in enumerate(clinician_histories)
            ]
            if clinician_remote is None:
                clinician_result = _generate_api_clinician_batch(
                    clinician_model, clinician_prompts, clinician_max_tokens
                )
            else:
                clinician_result = clinician_remote.generate_batch.remote(
                    clinician_prompts, clinician_max_tokens
                )
            clinician_texts = _validate_batch(
                clinician_result, len(questions), "clinician", exchange
            )

            patient_prompts: list[list[dict[str, str]]] = []
            for index, clinician_text in enumerate(clinician_texts):
                clinician_histories[index] = advance_clinician_history(
                    clinician_histories[index],
                    prior_patient_replies[index],
                    clinician_text,
                )
                patient_prompts.append(
                    patient_generation_messages(patient_histories[index], clinician_text)
                )
                turns[index].append(
                    {
                        "turn_index": 2 * exchange - 1,
                        "exchange_index": exchange,
                        "role": "clinician",
                        "content": clinician_text,
                        "model": clinician_result.get(
                            "model_versions", [clinician_model] * len(questions)
                        )[index],
                        "finish_reason": clinician_result["finish_reasons"][index],
                        "input_tokens": int(clinician_result["input_tokens"][index]),
                        "output_tokens": int(
                            clinician_result["output_tokens"][index]
                        ),
                        "latency_ms": int(clinician_result["latency_ms"][index]),
                    }
                )

            patient_result = patient.generate_batch.remote(
                patient_prompts, patient_max_tokens
            )
            patient_texts = _validate_batch(
                patient_result, len(questions), "patient", exchange
            )
            for index, patient_text in enumerate(patient_texts):
                patient_histories[index] = advance_patient_history(
                    patient_histories[index], clinician_texts[index], patient_text
                )
                prior_patient_replies[index] = patient_text
                turns[index].append(
                    {
                        "turn_index": 2 * exchange,
                        "exchange_index": exchange,
                        "role": "patient",
                        "content": patient_text,
                        "model": PATIENT_MODEL,
                        "finish_reason": patient_result["finish_reasons"][index],
                        "input_tokens": int(patient_result["input_tokens"][index]),
                        "output_tokens": int(patient_result["output_tokens"][index]),
                        "latency_ms": int(patient_result["latency_ms"][index]),
                    }
                )
            print(
                f"Batch {batch_number}/{batch_count}, exchange {exchange}/{exchanges}: "
                f"generated {len(questions)} clinician and patient turns"
            )
    except Exception as exc:
        return [
            generation_record(
                question=question,
                turns=turns[index],
                run_id=run_id,
                attempt=starting_attempts[question.question_id],
                exchanges=exchanges,
                patient_max_tokens=patient_max_tokens,
                clinician_max_tokens=clinician_max_tokens,
                clinician_model=clinician_model,
                clinician_temperature=clinician_temperature,
                status="failed",
                seed=SEED,
                error=exc,
            )
            for index, question in enumerate(questions)
        ]

    return [
        generation_record(
            question=question,
            turns=turns[index],
            run_id=run_id,
            attempt=starting_attempts[question.question_id],
            exchanges=exchanges,
            patient_max_tokens=patient_max_tokens,
            clinician_max_tokens=clinician_max_tokens,
            clinician_model=clinician_model,
            clinician_temperature=clinician_temperature,
            status="succeeded",
            seed=SEED,
        )
        for index, question in enumerate(questions)
    ]


@app.cls(
    image=vllm_image,
    gpu="H100",
    timeout=30 * MINUTES,
    startup_timeout=30 * MINUTES,
    scaledown_window=5 * MINUTES,
    volumes=volumes,
)
class PatientMistralSmall31:
    @modal.enter()
    def start(self) -> None:
        import vllm

        self.llm = vllm.LLM(
            model=PATIENT_MODEL,
            max_model_len=16384,
            gpu_memory_utilization=0.92,
            limit_mm_per_prompt={"image": 0, "video": 0},
            enforce_eager=True,
        )

    @modal.method()
    def generate_batch(
        self, conversations: list[list[dict[str, str]]], max_tokens: int
    ) -> dict[str, Any]:
        return _generate(
            self.llm,
            conversations,
            temperature=0.2,
            max_tokens=max_tokens,
        )


@app.cls(
    image=vllm_image,
    gpu="H100:2",
    timeout=30 * MINUTES,
    startup_timeout=30 * MINUTES,
    scaledown_window=5 * MINUTES,
    volumes=volumes,
)
class ClinicianQwen122B:
    @modal.enter()
    def start(self) -> None:
        import vllm

        self.llm = vllm.LLM(
            model=DEFAULT_CLINICIAN_MODEL,
            tensor_parallel_size=2,
            max_model_len=8192,
            gpu_memory_utilization=0.92,
            limit_mm_per_prompt={"image": 0, "video": 0},
            enforce_eager=True,
        )

    @modal.method()
    def generate_batch(
        self, conversations: list[list[dict[str, str]]], max_tokens: int
    ) -> dict[str, Any]:
        return _generate(
            self.llm,
            conversations,
            temperature=0.2,
            max_tokens=max_tokens,
        )


@app.cls(
    image=vllm_image,
    gpu="H100",
    timeout=30 * MINUTES,
    startup_timeout=30 * MINUTES,
    scaledown_window=5 * MINUTES,
    volumes=volumes,
)
class ClinicianQwen27B:
    @modal.enter()
    def start(self) -> None:
        import vllm

        self.llm = vllm.LLM(
            model=QWEN_27B_CLINICIAN_MODEL,
            max_model_len=8192,
            gpu_memory_utilization=0.92,
            limit_mm_per_prompt={"image": 0, "video": 0},
            enforce_eager=True,
        )

    @modal.method()
    def generate_batch(
        self, conversations: list[list[dict[str, str]]], max_tokens: int
    ) -> dict[str, Any]:
        return _generate(
            self.llm,
            conversations,
            temperature=0.2,
            max_tokens=max_tokens,
        )


def _modal_clinician(model: str) -> Any | None:
    provider, native_model = resolve_provider(model)
    if provider is not Provider.MODAL:
        return None
    if native_model == DEFAULT_CLINICIAN_MODEL:
        return ClinicianQwen122B()
    if native_model == QWEN_27B_CLINICIAN_MODEL:
        return ClinicianQwen27B()
    raise ValueError(
        "unsupported Modal clinician model for this runner: "
        f"{model}; choose {DEFAULT_CLINICIAN_MODEL} or "
        f"{QWEN_27B_CLINICIAN_MODEL}"
    )


@app.local_entrypoint()
def main(
    input_path: str = str(DEFAULT_INPUT),
    output_path: str = str(DEFAULT_OUTPUT),
    exchanges: int = DEFAULT_EXCHANGES,
    num_questions: int = 200,
    smoke_test: bool = False,
    patient_max_tokens: int = 160,
    clinician_max_tokens: int = DEFAULT_CLINICIAN_MAX_TOKENS,
    checkpoint_batch_size: int = 8,
    clinician_model: str = DEFAULT_CLINICIAN_MODEL,
    run_id: str = "",
    force: bool = False,
) -> None:
    if exchanges <= 0:
        raise ValueError("exchanges must be positive")
    if checkpoint_batch_size <= 0:
        raise ValueError("checkpoint_batch_size must be positive")
    if patient_max_tokens <= 0 or clinician_max_tokens <= 0:
        raise ValueError("output token limits must be positive")
    _load_dotenv()
    clinician_provider = resolve_provider(clinician_model)[0]
    clinician_temperature = 0.2 if clinician_provider is Provider.MODAL else None
    questions = load_questions(Path(input_path))
    if smoke_test:
        num_questions = 1
    if not 1 <= num_questions <= len(questions):
        raise ValueError(
            f"num_questions must be between 1 and {len(questions)}"
        )
    source = Path(input_path)
    destination = Path(output_path)
    resume = load_resume_state(destination)
    generation_keys = {
        question.question_id: build_generation_key(
            question.question_id,
            clinician_model=clinician_model,
            question_text_sha256=question.question_text_sha256,
            private_patient_context_sha256=question.private_patient_context_sha256,
            exchanges=exchanges,
            patient_max_tokens=patient_max_tokens,
            clinician_max_tokens=clinician_max_tokens,
            clinician_temperature=clinician_temperature,
            seed=SEED,
        )
        for question in questions[:num_questions]
    }
    pending = [
        question
        for question in questions[:num_questions]
        if force or generation_keys[question.question_id] not in resume.succeeded_keys
    ]
    skipped = num_questions - len(pending)
    if not pending:
        print("All selected MedSP1000 questions are already complete")
        return
    active_run_id = run_id or default_run_id()
    starting_attempts = {
        question.question_id: (
            resume.highest_attempt_by_key.get(generation_keys[question.question_id], 0)
            + 1
        )
        for question in pending
    }

    batches = list(chunked(pending, checkpoint_batch_size))
    clinician_remote = _modal_clinician(clinician_model)
    patient = PatientMistralSmall31()
    manifest_path = destination.parent / f"{active_run_id}.manifest.json"
    selected_question_ids = [
        question.question_id for question in questions[:num_questions]
    ]
    manifest_created_at = utc_now()
    succeeded = 0
    failed = 0
    checkpoint_count = 0

    def write_progress(run_status: str, error: BaseException | None = None) -> None:
        _write_manifest(
            manifest_path,
            run_id=active_run_id,
            input_path=source,
            output_path=destination,
            question_ids=selected_question_ids,
            exchanges=exchanges,
            patient_max_tokens=patient_max_tokens,
            clinician_max_tokens=clinician_max_tokens,
            clinician_model=clinician_model,
            clinician_temperature=clinician_temperature,
            force=force,
            batch_size=checkpoint_batch_size,
            planned=len(pending),
            skipped=skipped,
            succeeded=succeeded,
            failed=failed,
            checkpoint_count=checkpoint_count,
            run_status=run_status,
            created_at=manifest_created_at,
            error=error,
        )

    write_progress("running")
    try:
        for batch_number, question_batch in enumerate(batches, start=1):
            records = _generate_conversation_batch(
                questions=question_batch,
                clinician_remote=clinician_remote,
                clinician_model=clinician_model,
                clinician_temperature=clinician_temperature,
                patient=patient,
                run_id=active_run_id,
                starting_attempts=starting_attempts,
                exchanges=exchanges,
                patient_max_tokens=patient_max_tokens,
                clinician_max_tokens=clinician_max_tokens,
                batch_number=batch_number,
                batch_count=len(batches),
            )
            append_jsonl(destination, records)
            batch_succeeded = sum(
                record["status"] == "succeeded" for record in records
            )
            batch_failed = len(records) - batch_succeeded
            succeeded += batch_succeeded
            failed += batch_failed
            checkpoint_count += 1
            write_progress("running")
            print(
                f"Checkpoint {checkpoint_count}: durably saved {succeeded} succeeded and "
                f"{failed} failed attempts for {len(pending)} conversations to "
                f"{destination}"
            )
    except BaseException as exc:
        write_progress("interrupted", exc)
        raise

    write_progress("completed")
    print(
        f"Generation complete: {succeeded} succeeded, {failed} failed, "
        f"{skipped} skipped; "
        f"output={destination}; manifest={manifest_path}"
    )
