import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from generation import medsp1000, modal_medsp1000
from generation.medsp1000 import (
    EXPERIMENT_ID,
    PROMPT_VERSION,
    advance_clinician_history,
    advance_patient_history,
    append_jsonl,
    build_generation_key,
    chunked,
    clinician_generation_messages,
    clinician_messages,
    clinician_turn_prompt,
    generation_record,
    load_questions,
    load_resume_state,
    patient_generation_messages,
    patient_messages,
    patient_turn_prompt,
    project_root,
    transcript_text,
)
from inference import ModelResponse, Provider, TokenUsage


def _question_file(path: Path) -> Path:
    question_text = "A patient presents with neck pain."
    patient_context = "Opening statement: My neck hurts."
    record = {
        "schema_version": "2.0",
        "question_id": "question-1",
        "question_type": "multi_turn_standardized_patient",
        "question_text": question_text,
        "private_patient_context": patient_context,
        "source_scenario_path": "source/scenario1",
        "selection_reason": "test",
        "source_dataset": "byrLLCC/MedSP1000",
        "source_revision": "a" * 40,
        "cohort_index": 1,
        "selection_seed": 42,
        "quality_score": 10,
        "official_q100_member": False,
        "history_domains": ["present_illness"],
        "private_patient_context_sha256": hashlib.sha256(
            patient_context.encode()
        ).hexdigest(),
        "question_text_sha256": hashlib.sha256(question_text.encode()).hexdigest(),
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


def test_role_prompts_keep_material_separate(tmp_path: Path) -> None:
    question = load_questions(_question_file(tmp_path / "questions.jsonl"))[0]
    patient_prompt = patient_messages(question)[0]["content"]
    clinician_prompt = clinician_messages(question, 4)[0]["content"]
    assert "My neck hurts" in patient_prompt
    assert "A patient presents" not in patient_prompt
    assert "A patient presents" in clinician_prompt
    assert "My neck hurts" not in clinician_prompt
    assert "rubric" not in patient_prompt.lower()


def test_load_question_schema(tmp_path: Path) -> None:
    question = load_questions(_question_file(tmp_path / "questions.jsonl"))[0]
    assert question.question_id == "question-1"
    assert question.source_scenario_path == "source/scenario1"
    assert question.private_patient_context == "Opening statement: My neck hurts."
    assert question.question_text == "A patient presents with neck pain."


def test_load_question_rejects_changed_embedded_context(tmp_path: Path) -> None:
    path = _question_file(tmp_path / "questions.jsonl")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["private_patient_context"] += " Altered."
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="private_patient_context_sha256 mismatch"):
        load_questions(path)


def test_load_question_enforces_committed_schema(tmp_path: Path) -> None:
    path = _question_file(tmp_path / "questions.jsonl")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["schema_version"] = "1.0"
    record["unexpected"] = True
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema validation failed"):
        load_questions(path)


def test_clinician_turn_bounds() -> None:
    assert "turn 1 of 4" in clinician_turn_prompt(None, 1, 4)
    assert "patient reply" in clinician_turn_prompt("patient reply", 2, 4)
    with pytest.raises(ValueError):
        clinician_turn_prompt(None, 5, 4)


def test_patient_turn_prompt_preserves_clinician_message_and_repeats_grounding() -> None:
    prompt = patient_turn_prompt("Have you had a sore throat?")
    assert "Have you had a sore throat?" in prompt
    assert "suggested by the clinician are not patient facts" in prompt
    assert "If the material is silent" in prompt
    with pytest.raises(ValueError):
        patient_turn_prompt("  ")


def test_only_current_patient_turn_contains_grounding_reminder() -> None:
    base = [{"role": "system", "content": "patient packet"}]
    history = advance_patient_history(base, "First question", "First answer")
    prompt = patient_generation_messages(history, "Second question")
    assert history == [
        {"role": "system", "content": "patient packet"},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
    ]
    assert sum("silently check" in message["content"] for message in prompt) == 1
    assert "Second question" in prompt[-1]["content"]


def test_only_current_clinician_turn_contains_turn_control() -> None:
    base = [{"role": "system", "content": "clinician packet"}]
    history = advance_clinician_history(base, None, "Opening question")
    prompt = clinician_generation_messages(history, "Patient answer", 2, 4)
    assert "turn 1" not in " ".join(message["content"] for message in history)
    assert sum("Continue the consultation" in message["content"] for message in prompt) == 1
    assert prompt[-1]["content"].startswith("Patient answer")


def test_generation_key_changes_with_prompt_or_run_configuration() -> None:
    default_key = build_generation_key("question-1")
    assert PROMPT_VERSION in default_key
    assert "patient-max-320" in default_key
    assert default_key != build_generation_key("question-1", exchanges=5)
    assert default_key != build_generation_key("question-1", seed=7)
    assert default_key != build_generation_key(
        "question-1", clinician_model="claude-sonnet-5"
    )
    assert default_key != build_generation_key(
        "question-1", clinician_temperature=None
    )
    assert default_key != build_generation_key(
        "question-1",
        clinician_reasoning_mode="reasoning",
        clinician_reasoning_effort="medium",
    )
    first_inputs = build_generation_key(
        "question-1",
        question_text_sha256="a" * 64,
        private_patient_context_sha256="b" * 64,
    )
    second_inputs = build_generation_key(
        "question-1",
        question_text_sha256="c" * 64,
        private_patient_context_sha256="b" * 64,
    )
    assert first_inputs != second_inputs


def test_chunked_preserves_order_and_bounds_checkpoint_loss() -> None:
    assert list(chunked(list(range(10)), 4)) == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]
    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(chunked([1], 0))


def test_project_root_tolerates_modal_shallow_mount() -> None:
    assert project_root(Path("/root/modal_medsp1000.py")) == Path("/root")


def test_append_jsonl_syncs_each_complete_record(
    tmp_path: Path, monkeypatch
) -> None:
    sync_calls: list[int] = []
    lock_calls: list[int] = []
    monkeypatch.setattr(medsp1000.os, "fsync", sync_calls.append)
    monkeypatch.setattr(
        medsp1000.fcntl,
        "flock",
        lambda _file_descriptor, operation: lock_calls.append(operation),
    )
    output = tmp_path / "output.jsonl"
    append_jsonl(output, [{"id": 1}, {"id": 2}])
    assert sync_calls and len(sync_calls) == 2
    assert lock_calls == [
        medsp1000.fcntl.LOCK_EX,
        medsp1000.fcntl.LOCK_UN,
        medsp1000.fcntl.LOCK_EX,
        medsp1000.fcntl.LOCK_UN,
    ]
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"id": 1},
        {"id": 2},
    ]


def test_api_clinician_batch_normalizes_provider_outputs(monkeypatch) -> None:
    def fake_call_model(model, messages, *, max_output_tokens, **provider_options):
        assert model == "claude-test"
        assert max_output_tokens == 220
        assert provider_options == {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
        }
        text = f"Reply to {messages[-1]['content']}"
        return ModelResponse(
            text=text,
            provider=Provider.ANTHROPIC,
            model="claude-test-20260824",
            request_id=f"request-{messages[-1]['content']}",
            finish_reason="end_turn",
            usage=TokenUsage(input_tokens=12, output_tokens=5),
            raw=SimpleNamespace(
                content=[SimpleNamespace(type="text", text=text)]
            ),
        )

    monkeypatch.setattr(modal_medsp1000, "call_model", fake_call_model)
    result = modal_medsp1000._generate_api_clinician_batch(
        "claude-test",
        [
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ],
        220,
    )
    assert result["texts"] == ["Reply to first", "Reply to second"]
    assert result["finish_reasons"] == ["end_turn", "end_turn"]
    assert result["model_versions"] == [
        "claude-test-20260824",
        "claude-test-20260824",
    ]
    assert result["input_tokens"] == [12, 12]
    assert result["output_tokens"] == [5, 5]
    assert result["provider_request_ids"] == ["request-first", "request-second"]
    assert result["provider_output_item_counts"] == [1, 1]
    assert result["provider_text_block_counts"] == [1, 1]
    assert result["provider_text_block_sha256"] == [
        [hashlib.sha256(b"Reply to first").hexdigest()],
        [hashlib.sha256(b"Reply to second").hexdigest()],
    ]


def test_duplicate_quality_flag_does_not_rewrite_text() -> None:
    text = "What brings you in today?What brings you in today?"
    assert modal_medsp1000._quality_flags(text) == ["exact_repeated_message"]
    assert text == "What brings you in today?What brings you in today?"


def test_openai_multi_message_blocks_are_hashed_separately() -> None:
    repeated = "What brings you in today?"
    response = ModelResponse(
        text=repeated + repeated,
        provider=Provider.OPENAI,
        model="gpt-test",
        request_id="response-1",
        raw=SimpleNamespace(
            output=[
                SimpleNamespace(type="reasoning"),
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text=repeated)],
                ),
                SimpleNamespace(type="reasoning"),
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text=repeated)],
                ),
            ]
        ),
    )
    diagnostics = modal_medsp1000._provider_response_diagnostics(response)
    repeated_hash = hashlib.sha256(repeated.encode()).hexdigest()
    assert diagnostics == {
        "provider_request_id": "response-1",
        "provider_output_item_count": 4,
        "provider_text_block_count": 2,
        "provider_text_block_sha256": [repeated_hash, repeated_hash],
    }


def test_api_clinician_failure_is_isolated_within_batch(monkeypatch) -> None:
    def fake_call_model(model, messages, *, max_output_tokens, **provider_options):
        del model, max_output_tokens, provider_options
        if messages[-1]["content"] == "second":
            raise RuntimeError("provider unavailable")
        return ModelResponse(
            text="First reply",
            provider=Provider.ANTHROPIC,
            model="claude-test",
            finish_reason="end_turn",
            usage=TokenUsage(input_tokens=12, output_tokens=3),
        )

    monkeypatch.setattr(modal_medsp1000, "call_model", fake_call_model)
    result = modal_medsp1000._generate_api_clinician_batch(
        "claude-test",
        [
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ],
        220,
    )
    assert result["texts"] == ["First reply", ""]
    assert result["errors"][0] is None
    assert isinstance(result["errors"][1], RuntimeError)
    assert str(result["errors"][1]) == "provider unavailable"


@pytest.mark.parametrize(
    ("model", "mode", "provider_options"),
    [
        (
            "gpt-5.6-sol",
            "reasoning",
            {"reasoning": {"effort": "medium"}},
        ),
        (
            "claude-opus-5",
            "adaptive",
            {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "medium"},
            },
        ),
        (
            "gemini-3.1-pro-preview",
            "thinking_level",
            {"thinking_config": {"thinking_level": "medium"}},
        ),
    ],
)
def test_api_reasoning_is_explicitly_medium(
    model: str, mode: str, provider_options: dict[str, object]
) -> None:
    config = modal_medsp1000._clinician_reasoning(model)
    assert config == {
        "enabled": True,
        "mode": mode,
        "effort": "medium",
        "provider_options": provider_options,
    }


def test_batch_failure_returns_auditable_failed_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    question = load_questions(_question_file(tmp_path / "questions.jsonl"))[0]
    monkeypatch.setattr(
        modal_medsp1000,
        "_generate_api_clinician_batch",
        lambda *args: {
            "texts": ["Where does it hurt?"],
            "input_tokens": [10],
            "output_tokens": [4],
            "finish_reasons": ["stop"],
            "model_versions": ["clinician-model-version"],
            "latency_ms": [100],
        },
    )
    patient = SimpleNamespace(
        generate_batch=SimpleNamespace(
            remote=lambda *args: {
                "texts": ["My neck hurts."],
                "input_tokens": [20],
                "output_tokens": [4],
                "finish_reasons": ["length"],
                "latency_ms": [50],
            }
        )
    )
    records = modal_medsp1000._generate_conversation_batch(
        questions=[question],
        clinician_remote=None,
        clinician_model="claude-test",
        clinician_temperature=None,
        patient=patient,
        run_id="run-1",
        starting_attempts={question.question_id: 2},
        exchanges=1,
        patient_max_tokens=160,
        clinician_max_tokens=220,
        batch_number=1,
        batch_count=1,
    )
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert records[0]["attempt"] == 2
    assert records[0]["turn_count"] == 1
    assert records[0]["error_type"] == "RuntimeError"
    assert "truncated patient response" in records[0]["error_message"]


def test_patient_failure_is_isolated_within_batch(tmp_path: Path, monkeypatch) -> None:
    first = load_questions(_question_file(tmp_path / "questions.jsonl"))[0]
    second = replace(first, question_id="question-2")

    def clinician_batch(_model, conversations, _max_tokens):
        count = len(conversations)
        return {
            "texts": ["Where does it hurt?"] * count,
            "input_tokens": [10] * count,
            "output_tokens": [4] * count,
            "finish_reasons": ["stop"] * count,
            "model_versions": ["clinician-version"] * count,
            "latency_ms": [100] * count,
            "provider_request_ids": [f"request-{index}" for index in range(count)],
            "provider_output_item_counts": [2] * count,
            "provider_text_block_counts": [1] * count,
            "provider_text_block_sha256": [["a" * 64]] * count,
        }

    monkeypatch.setattr(
        modal_medsp1000,
        "_generate_api_clinician_batch",
        clinician_batch,
    )
    patient_calls = 0

    def patient_batch(*args):
        nonlocal patient_calls
        patient_calls += 1
        count = len(args[0])
        return {
            "texts": ["My neck hurts."] * count,
            "input_tokens": [20] * count,
            "output_tokens": [4] * count,
            "finish_reasons": (
                ["stop", "length"] if patient_calls == 1 else ["stop"]
            ),
            "latency_ms": [50] * count,
        }

    patient = SimpleNamespace(
        generate_batch=SimpleNamespace(
            remote=patient_batch
        )
    )
    records = modal_medsp1000._generate_conversation_batch(
        questions=[first, second],
        clinician_remote=None,
        clinician_model="claude-test",
        clinician_temperature=None,
        patient=patient,
        run_id="run-1",
        starting_attempts={first.question_id: 1, second.question_id: 1},
        exchanges=2,
        patient_max_tokens=160,
        clinician_max_tokens=220,
        batch_number=1,
        batch_count=1,
    )
    assert [record["status"] for record in records] == ["succeeded", "failed"]
    assert [record["turn_count"] for record in records] == [4, 1]
    assert patient_calls == 2
    assert records[0]["turns"][0]["provider_request_id"] == "request-0"
    assert records[0]["turns"][0]["provider_output_item_count"] == 2
    assert records[0]["turns"][0]["provider_text_block_sha256"] == ["a" * 64]
    assert records[1]["error_message"] == (
        "truncated patient response at exchange 1: length"
    )


def test_resume_and_transcript(tmp_path: Path) -> None:
    output = tmp_path / "generations.jsonl"
    question = load_questions(_question_file(tmp_path / "questions.jsonl"))[0]
    turns = [
        {
            "turn_index": 1,
            "exchange_index": 1,
            "role": "clinician",
            "content": "Hello",
            "model": "clinician-model",
            "finish_reason": "stop",
            "input_tokens": 10,
            "output_tokens": 2,
            "latency_ms": 100,
        },
        {
            "turn_index": 2,
            "exchange_index": 1,
            "role": "patient",
            "content": "Hi",
            "model": "patient-model",
            "finish_reason": "stop",
            "input_tokens": 20,
            "output_tokens": 1,
            "latency_ms": 50,
        },
    ]
    record = generation_record(
        question=question,
        turns=turns,
        run_id="run-1",
        attempt=2,
        exchanges=1,
        patient_max_tokens=160,
        clinician_max_tokens=220,
        status="succeeded",
        seed=20260824,
    )
    generation_key = record["generation_key"]
    output.write_text(json.dumps(record) + "\n", encoding="utf-8")
    resume = load_resume_state(output)
    assert resume.succeeded_keys == {generation_key}
    assert resume.highest_attempt_by_key == {generation_key: 2}
    assert transcript_text(
        [
            {"role": "clinician", "content": "Hello"},
            {"role": "patient", "content": "Hi"},
        ]
    ) == "CLINICIAN: Hello\nPATIENT: Hi"


def test_resume_rejects_cross_field_corruption(tmp_path: Path) -> None:
    question = load_questions(_question_file(tmp_path / "questions.jsonl"))[0]
    record = generation_record(
        question=question,
        turns=[],
        run_id="run-1",
        attempt=1,
        exchanges=1,
        patient_max_tokens=160,
        clinician_max_tokens=220,
        status="failed",
        seed=42,
        error=RuntimeError("provider failed"),
    )
    record["input_tokens"] = 1
    output = tmp_path / "generations.jsonl"
    output.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input_tokens does not match turns"):
        load_resume_state(output)


def test_multiturn_output_matches_schema_and_aggregates_usage(tmp_path: Path) -> None:
    question = load_questions(_question_file(tmp_path / "questions.jsonl"))[0]
    turns = [
        {
            "turn_index": 1,
            "exchange_index": 1,
            "role": "clinician",
            "content": "Hello",
            "model": "clinician-model",
            "finish_reason": "stop",
            "input_tokens": 100,
            "output_tokens": 10,
            "latency_ms": 900,
        },
        {
            "turn_index": 2,
            "exchange_index": 1,
            "role": "patient",
            "content": "My neck hurts.",
            "model": "patient-model",
            "finish_reason": "stop",
            "input_tokens": 200,
            "output_tokens": 6,
            "latency_ms": 400,
        },
    ]
    record = generation_record(
        question=question,
        turns=turns,
        run_id="medsp1000-test-run",
        attempt=1,
        exchanges=1,
        patient_max_tokens=160,
        clinician_max_tokens=220,
        status="succeeded",
        seed=42,
    )
    schema = json.loads(
        Path("data/outputs/schemas/medsp1000_multiturn_generation.schema.json").read_text(
            encoding="utf-8"
        )
    )
    medsp1000.validate_record(record, medsp1000.OUTPUT_SCHEMA, location="test record")
    assert set(record) == set(schema["properties"])
    assert record["turns"] == turns
    assert record["transcript_text"] == "CLINICIAN: Hello\nPATIENT: My neck hurts."
    assert record["input_tokens"] == 300
    assert record["output_tokens"] == 16
    assert record["latency_ms"] == 1300
    assert record["clinician_input_tokens"] == 100
    assert record["patient_input_tokens"] == 200
