from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from inference import (
    Message,
    ModalConfig,
    Provider,
    ProviderNotFoundError,
    call_model,
    resolve_provider,
)


class Answer(BaseModel):
    answer: str


class FakeResponses:
    def __init__(self, response: object) -> None:
        self.response = response
        self.method: str | None = None
        self.request: dict = {}

    def create(self, **request: object) -> object:
        self.method = "create"
        self.request = request
        return self.response

    def parse(self, **request: object) -> object:
        self.method = "parse"
        self.request = request
        return self.response


class FakeMessages(FakeResponses):
    pass


def test_routes_known_models_and_explicit_prefixes() -> None:
    assert resolve_provider("gpt-5.5") == (Provider.OPENAI, "gpt-5.5")
    assert resolve_provider("claude-sonnet-5") == (
        Provider.ANTHROPIC,
        "claude-sonnet-5",
    )
    assert resolve_provider("gemini-3.1-flash-lite") == (
        Provider.GEMINI,
        "gemini-3.1-flash-lite",
    )
    assert resolve_provider("modal/Qwen3.6-35B") == (Provider.MODAL, "Qwen3.6-35B")


def test_unknown_model_requires_an_explicit_provider() -> None:
    with pytest.raises(ProviderNotFoundError):
        resolve_provider("custom-model")
    assert resolve_provider("custom-model", "modal") == (
        Provider.MODAL,
        "custom-model",
    )


def test_openai_translates_messages_and_parses_pydantic() -> None:
    raw = SimpleNamespace(
        output_text='{"answer":"yes"}',
        model="gpt-test-version",
        id="resp_1",
        status="completed",
        usage=SimpleNamespace(input_tokens=10, output_tokens=4),
    )
    responses = FakeResponses(raw)
    client = SimpleNamespace(responses=responses)

    result = call_model(
        "openai/gpt-test",
        [
            Message("system", "Be concise."),
            Message("user", "Is this working?"),
        ],
        response_format=Answer,
        client=client,
        max_output_tokens=50,
    )

    assert responses.method == "parse"
    assert responses.request == {
        "model": "gpt-test",
        "input": [{"role": "user", "content": "Is this working?"}],
        "instructions": "Be concise.",
        "max_output_tokens": 50,
        "text_format": Answer,
    }
    assert result.parsed == Answer(answer="yes")
    assert result.usage.total_tokens == 14


def test_anthropic_uses_output_config_for_raw_json_schema() -> None:
    raw = SimpleNamespace(
        content=[SimpleNamespace(type="text", text='{"answer":"yes"}')],
        model="claude-test-version",
        id="msg_1",
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=8, output_tokens=3),
    )
    messages = FakeMessages(raw)
    client = SimpleNamespace(messages=messages)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    result = call_model(
        "anthropic/claude-test",
        "Is this working?",
        response_format=schema,
        client=client,
    )

    assert messages.method == "create"
    assert messages.request["output_config"] == {
        "format": {"type": "json_schema", "schema": schema}
    }
    assert result.parsed == {"answer": "yes"}


def test_gemini_translates_roles_config_and_usage() -> None:
    raw = SimpleNamespace(
        text='{"answer":"yes"}',
        model_version="gemini-test-version",
        response_id="gemini_1",
        candidates=[SimpleNamespace(finish_reason="STOP")],
        usage_metadata=SimpleNamespace(prompt_token_count=7, candidates_token_count=3),
    )

    class FakeModels:
        def __init__(self) -> None:
            self.request: dict = {}

        def generate_content(self, **request: object) -> object:
            self.request = request
            return raw

    models = FakeModels()
    client = SimpleNamespace(models=models)

    result = call_model(
        "google/gemini-test",
        [
            Message("user", "Question"),
            Message("assistant", "First answer"),
            Message("user", "Return JSON now"),
        ],
        system="Be concise.",
        response_format=Answer,
        client=client,
        temperature=0.2,
    )

    contents = models.request["contents"]
    assert [content.role for content in contents] == ["user", "model", "user"]
    assert [content.parts[0].text for content in contents] == [
        "Question",
        "First answer",
        "Return JSON now",
    ]
    assert models.request["config"] == {
        "temperature": 0.2,
        "system_instruction": "Be concise.",
        "max_output_tokens": 1024,
        "response_mime_type": "application/json",
        "response_json_schema": Answer.model_json_schema(),
    }
    assert result.parsed == Answer(answer="yes")
    assert result.usage.total_tokens == 10


class FakeRemoteFunction:
    def __init__(self, result: object) -> None:
        self.result = result
        self.request: dict = {}

    def remote(self, **request: object) -> object:
        self.request = request
        return self.result


def test_modal_contract_and_normalized_result() -> None:
    function = FakeRemoteFunction(
        {
            "text": json.dumps({"answer": "yes"}),
            "model": "Qwen-test-revision",
            "request_id": "modal_1",
            "finish_reason": "stop",
            "usage": {"input_tokens": 9, "output_tokens": 4},
        }
    )

    result = call_model(
        "modal/Qwen-test",
        "Is this working?",
        system="Be concise.",
        response_format=Answer,
        remote_function=function,
    )

    assert function.request["model"] == "Qwen-test"
    assert function.request["messages"] == [
        {"role": "user", "content": "Is this working?"}
    ]
    assert function.request["system"] == "Be concise."
    assert function.request["response_schema"] == Answer.model_json_schema()
    assert result.parsed == Answer(answer="yes")
    assert result.request_id == "modal_1"


def test_modal_openai_web_transport() -> None:
    raw = SimpleNamespace(
        id="chatcmpl_1",
        model="qwen2.5-3b",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"answer":"yes"}'),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=5),
    )
    completions = FakeResponses(raw)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = call_model(
        "modal/qwen2.5-3b",
        "Is this working?",
        system="Be concise.",
        response_format=Answer,
        modal_config=ModalConfig(
            app_name="vbench-vllm-qwen2-5-3b",
            function_name="serve",
            transport="openai_web",
        ),
        client=client,
    )

    assert completions.request["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Is this working?"},
    ]
    assert completions.request["response_format"]["json_schema"]["schema"] == (
        Answer.model_json_schema()
    )
    assert result.parsed == Answer(answer="yes")
    assert result.usage.total_tokens == 16


def test_conflicting_prefix_and_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="model prefix selects openai"):
        resolve_provider("openai/gpt-test", "anthropic")
