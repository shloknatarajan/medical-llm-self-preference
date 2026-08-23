# Unified inference

`call_model` routes a model name to OpenAI, Anthropic, Gemini, or a deployed
Modal function and returns the same `ModelResponse` shape for every backend.

```python
from pydantic import BaseModel

from inference import ModalConfig, call_model


class Judgment(BaseModel):
    winner: str
    reasoning: str


# Provider inferred from the model family.
text_response = call_model(
    "claude-sonnet-5",
    "Explain the result.",
    system="You are a careful clinical evaluator.",
)
print(text_response.text)

# Pydantic types enable provider-native structured output and local validation.
judgment_response = call_model(
    "gpt-5.5",
    "Compare response A and response B.",
    response_format=Judgment,
)
print(judgment_response.parsed.winner)

# Raw JSON Schema is also accepted; `parsed` will be a dict/list.
schema_response = call_model(
    "gemini-3.1-flash-lite",
    "Return a verdict.",
    response_format={
        "type": "object",
        "properties": {"verdict": {"type": "string"}},
        "required": ["verdict"],
        "additionalProperties": False,
    },
)

# Modal models can use a prefix or provider="modal". The target must already be
# deployed and implement the contract in providers/modal.py.
modal_response = call_model(
    "modal/Qwen3.6-35B",
    "Explain the result.",
    modal_config=ModalConfig(
        app_name="medical-llm-inference",
        function_name="generate",
    ),
)

# Modal-hosted vLLM/OpenAI-compatible web endpoints use the web transport.
vllm_response = call_model(
    "modal/qwen2.5-3b",
    "Explain the result.",
    modal_config=ModalConfig(
        app_name="vbench-vllm-qwen2-5-3b",
        function_name="serve",
        transport="openai_web",
    ),
)
```

Credentials are read by the provider SDKs from `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, and Modal's standard token/profile
configuration. A provider can be forced with `provider="openai"`,
`"anthropic"`, `"gemini"`, or `"modal"`.
