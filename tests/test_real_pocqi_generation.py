from __future__ import annotations

import json
import os
from pathlib import Path

from generation.generate_real_pocqi import (
    infer_generator_family,
    load_dotenv,
    parse_args,
    run,
)
from generation.real_pocqi import GenerationStatus, RealPocqiOutput
from inference import ModelResponse, Provider, TokenUsage


def write_question(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "question_id": "question-1",
                "question_text": "What is the treatment?",
                "specialty": "Cardiology",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_model_family_is_independent_of_modal_transport() -> None:
    assert infer_generator_family("gpt-5.5") == "openai"
    assert infer_generator_family("claude-sonnet-5") == "anthropic"
    assert infer_generator_family("gemini-3.1-flash-lite") == "google"
    assert infer_generator_family("modal/Qwen3.6-35B") == "qwen"


def test_pipeline_persists_retries_and_resumes(tmp_path: Path) -> None:
    input_path = tmp_path / "questions.jsonl"
    output_path = tmp_path / "outputs" / "generations.jsonl"
    write_question(input_path)
    calls = 0

    def fake_model_caller(*args: object, **kwargs: object) -> ModelResponse[object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("temporary timeout")
        return ModelResponse(
            text="Use guideline-directed treatment.",
            provider=Provider.OPENAI,
            model="gpt-5.5-2026-08-01",
            request_id="request-1",
            finish_reason="completed",
            usage=TokenUsage(input_tokens=100, output_tokens=20),
        )

    args = parse_args(
        [
            "--models",
            "gpt-5.5",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--run-id",
            "test-run-1",
            "--num-questions",
            "1",
            "--no-shuffle",
            "--max-concurrency",
            "1",
            "--retries",
            "1",
            "--retry-delay-seconds",
            "0",
        ]
    )
    assert run(args, model_caller=fake_model_caller) == 0

    records = [
        RealPocqiOutput.from_json(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record.status for record in records] == [
        GenerationStatus.FAILED,
        GenerationStatus.SUCCEEDED,
    ]
    assert [record.attempt for record in records] == [1, 2]
    assert records[0].generation_key == records[1].generation_key
    assert records[0].generation_id != records[1].generation_id
    assert records[1].response_text == "Use guideline-directed treatment."
    assert records[1].input_tokens == 100
    assert records[1].output_tokens == 20

    resumed_args = parse_args(
        [
            "--models",
            "gpt-5.5",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--run-id",
            "test-run-2",
            "--num-questions",
            "1",
            "--no-shuffle",
        ]
    )
    assert run(resumed_args, model_caller=fake_model_caller) == 0
    assert calls == 2
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 2

    manifest = json.loads(
        (output_path.parent / "test-run-2.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["logical_generations"] == {
        "planned": 0,
        "skipped_existing": 1,
        "succeeded": 0,
        "failed": 0,
    }


def test_pipeline_omits_temperature_by_default(tmp_path: Path) -> None:
    input_path = tmp_path / "questions.jsonl"
    output_path = tmp_path / "generations.jsonl"
    write_question(input_path)
    captured_kwargs: dict[str, object] = {}

    def fake_model_caller(*args: object, **kwargs: object) -> ModelResponse[object]:
        captured_kwargs.update(kwargs)
        return ModelResponse(
            text="Use guideline-directed treatment.",
            provider=Provider.ANTHROPIC,
            model="claude-sonnet-5",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    args = parse_args(
        [
            "--models",
            "claude-sonnet-5",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--run-id",
            "no-temperature-run",
            "--num-questions",
            "1",
            "--no-shuffle",
        ]
    )
    assert run(args, model_caller=fake_model_caller) == 0
    assert "temperature" not in captured_kwargs
    record = RealPocqiOutput.from_json(output_path.read_text(encoding="utf-8"))
    assert record.temperature is None
    assert record.max_output_tokens == 4096


def test_pipeline_rejects_truncated_response_without_retrying(tmp_path: Path) -> None:
    input_path = tmp_path / "questions.jsonl"
    output_path = tmp_path / "generations.jsonl"
    write_question(input_path)
    calls = 0

    def fake_model_caller(*args: object, **kwargs: object) -> ModelResponse[object]:
        nonlocal calls
        calls += 1
        return ModelResponse(
            text="Partial answer",
            provider=Provider.OPENAI,
            model="gpt-5.6-sol",
            request_id="response-1",
            finish_reason="incomplete",
            usage=TokenUsage(input_tokens=20, output_tokens=4096),
        )

    args = parse_args(
        [
            "--models",
            "gpt-5.6-sol",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--run-id",
            "truncated-run",
            "--num-questions",
            "1",
            "--no-shuffle",
            "--retries",
            "2",
        ]
    )
    assert run(args, model_caller=fake_model_caller) == 1
    assert calls == 1
    record = RealPocqiOutput.from_json(output_path.read_text(encoding="utf-8"))
    assert record.status is GenerationStatus.FAILED
    assert record.response_text == "Partial answer"
    assert record.error_type == "TruncatedResponseError"
    assert record.finish_reason == "incomplete"
    assert record.output_tokens == 4096


def test_pipeline_queues_models_round_robin_by_question(tmp_path: Path) -> None:
    input_path = tmp_path / "questions.jsonl"
    output_path = tmp_path / "generations.jsonl"
    input_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "question_id": f"question-{index}",
                    "question_text": f"Question {index}?",
                    "specialty": "Cardiology",
                }
            )
            for index in (1, 2)
        )
        + "\n",
        encoding="utf-8",
    )
    call_order: list[tuple[str, str]] = []

    def fake_model_caller(
        model: str, user_prompt: str, **kwargs: object
    ) -> ModelResponse[object]:
        question_id = "question-1" if "Question 1?" in user_prompt else "question-2"
        call_order.append((question_id, model))
        return ModelResponse(
            text="Complete answer",
            provider=Provider.OPENAI,
            model=model,
        )

    args = parse_args(
        [
            "--models",
            "gpt-5.6-sol",
            "claude-opus-5",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--run-id",
            "round-robin-run",
            "--num-questions",
            "2",
            "--no-shuffle",
            "--max-concurrency",
            "1",
        ]
    )
    assert run(args, model_caller=fake_model_caller) == 0
    assert call_order == [
        ("question-1", "gpt-5.6-sol"),
        ("question-1", "claude-opus-5"),
        ("question-2", "gpt-5.6-sol"),
        ("question-2", "claude-opus-5"),
    ]


def test_load_dotenv_preserves_exported_values(tmp_path: Path, monkeypatch) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "EXISTING_KEY=from-file\nexport NEW_KEY='new value'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EXISTING_KEY", "from-shell")
    monkeypatch.delenv("NEW_KEY", raising=False)

    load_dotenv(dotenv_path)

    assert os.environ["EXISTING_KEY"] == "from-shell"
    assert os.environ["NEW_KEY"] == "new value"


def test_pipeline_records_terminal_failure(tmp_path: Path) -> None:
    input_path = tmp_path / "questions.jsonl"
    output_path = tmp_path / "generations.jsonl"
    write_question(input_path)

    def failing_model_caller(*args: object, **kwargs: object) -> ModelResponse[object]:
        raise RuntimeError("provider unavailable")

    args = parse_args(
        [
            "--models",
            "gpt-5.5",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--run-id",
            "failed-run",
            "--num-questions",
            "1",
            "--no-shuffle",
            "--retries",
            "0",
        ]
    )
    assert run(args, model_caller=failing_model_caller) == 1
    record = RealPocqiOutput.from_json(output_path.read_text(encoding="utf-8"))
    assert record.status is GenerationStatus.FAILED
    assert record.error_type == "RuntimeError"
    assert record.error_message == "provider unavailable"
