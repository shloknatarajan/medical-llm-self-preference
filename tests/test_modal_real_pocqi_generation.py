from pathlib import Path

from generation.modal_real_pocqi_generation import _truncated_keys_below_cap
from generation.real_pocqi import GenerationStatus, RealPocqiOutput


def _record(*, attempt: int, finish_reason: str, cap: int) -> RealPocqiOutput:
    return RealPocqiOutput(
        experiment_id="experiment",
        run_id="run",
        generation_key="key",
        generation_id=f"generation-{attempt}",
        attempt=attempt,
        status=GenerationStatus.SUCCEEDED,
        created_at="2026-08-24T00:00:00Z",
        question_id="question",
        question_text="Question?",
        specialty="Medicine",
        generator_family="qwen",
        generator_model="Qwen/test",
        prompt_template_id="prompt",
        system_prompt="System",
        user_prompt="User",
        response_text="Answer",
        finish_reason=finish_reason,
        max_output_tokens=cap,
    )


def test_truncated_keys_below_cap_uses_latest_successful_attempt(tmp_path: Path) -> None:
    output = tmp_path / "output.jsonl"
    output.write_text(
        _record(attempt=1, finish_reason="length", cap=1024).to_json()
        + "\n"
        + _record(attempt=2, finish_reason="stop", cap=2048).to_json()
        + "\n",
        encoding="utf-8",
    )

    assert _truncated_keys_below_cap(output, "experiment", 2048) == frozenset()


def test_truncated_keys_below_cap_selects_smaller_capped_attempt(tmp_path: Path) -> None:
    output = tmp_path / "output.jsonl"
    output.write_text(
        _record(attempt=1, finish_reason="length", cap=1024).to_json() + "\n",
        encoding="utf-8",
    )

    assert _truncated_keys_below_cap(output, "experiment", 2048) == {"key"}
