"""Judge MedSP1000 clinician trajectories with rubric scoring and ranking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path

from inference import call_model

from .judge_real_pocqi import (
    JudgingCondition,
    JudgingTaskProfile,
    ModelCaller,
    PocqiJudgingSettings,
    PocqiResponseInput,
    PocqiResumeTracker,
    build_judgment_keys,
    judge_ranked_responses,
)
from .real_pocqi import (
    PocqiJudgingCase,
    PocqiJudgmentRecord,
    PocqiResponseCandidate,
    RubricAndModelRankingOutput,
)


RUBRIC_AND_RANKING_PROMPT_ID = "medsp1000_rubric_and_model_ranking_v1"
MEDSP1000_JUDGING_CASES = (PocqiJudgingCase.RUBRIC_AND_MODEL_RANKING,)

RUBRIC_SYSTEM_PROMPT = """You are an expert US physician evaluating multi-turn clinician trajectories from the same standardized-patient scenario.

Give each trajectory an absolute score from 0 to 5 on each axis, where 0 is the lowest possible performance and 5 is the highest:
- accuracy: How factually and clinically accurate are the clinician's statements across the encounter?
- clinical_utility: How useful is the clinician's overall interaction for providing high-quality clinical care?
- source_quality: How well grounded are the clinician's claims and recommendations in authoritative clinical knowledge? Do not require formal citations in dialogue.
- verifiability: How specific, transparent, and checkable are the clinician's claims, reasoning, and recommendations?
- completeness: How completely does the clinician address the scenario using the information available over the full interaction?

Score every trajectory independently and evaluate only the clinician's behavior. Patient turns provide conversational context and evidence about what information was available; do not score the patient. Use only the scenario initialization and candidate trajectories. Do not infer or identify which model produced a trajectory."""

RUBRIC_AND_RANKING_INSTRUCTION = (
    "First score every candidate clinician trajectory on all five rubric "
    "axes. Then rank all trajectories from best-performing to "
    "worst-performing using your overall clinical judgment. The ranking does "
    "not need to follow the score sums. Return every response ID exactly once."
)


def _user_prompt(
    scenario_text: str,
    scenario_path: str,
    candidates: Sequence[PocqiResponseCandidate],
    trajectory_texts: Mapping[str, str],
    instruction: str,
    *,
    view_turn_count: int | None = None,
) -> str:
    rendered = "\n\n".join(
        f'<candidate_trajectory id="{candidate.response_id}">\n'
        f"{trajectory_texts[candidate.response_id]}\n"
        "</candidate_trajectory>"
        for candidate in candidates
    )
    view_notice = (
        ""
        if view_turn_count is None
        else (
            "\n\nTRAJECTORY VIEW:\n"
            f"Each candidate contains only its first {view_turn_count} role turns. "
            "Evaluate only the interaction shown."
        )
    )
    return f"""SOURCE SCENARIO:
{scenario_path}

CLINICIAN INITIALIZATION:
{scenario_text}{view_notice}

CANDIDATE TRAJECTORIES:
{rendered}

TASK:
{instruction}"""


MEDSP1000_JUDGING_PROFILE = JudgingTaskProfile(
    conditions=(
        JudgingCondition(
            case=PocqiJudgingCase.RUBRIC_AND_MODEL_RANKING,
            prompt_template_id=RUBRIC_AND_RANKING_PROMPT_ID,
            system_prompt=RUBRIC_SYSTEM_PROMPT,
            instruction=RUBRIC_AND_RANKING_INSTRUCTION,
            output_type=RubricAndModelRankingOutput,
        ),
    ),
    user_prompt_builder=_user_prompt,
)


def medsp1000_judging_profile(
    view_turn_count: int | None = None,
) -> JudgingTaskProfile:
    """Return the full-trajectory or prefix-view judging profile."""

    if view_turn_count is None:
        return MEDSP1000_JUDGING_PROFILE
    if view_turn_count <= 0 or view_turn_count % 2:
        raise ValueError("view_turn_count must be a positive even integer")

    conditions = tuple(
        JudgingCondition(
            case=condition.case,
            prompt_template_id=(
                condition.prompt_template_id.removesuffix("_v1")
                + f"_first_{view_turn_count}_turns_v1"
            ),
            system_prompt=condition.system_prompt.replace(
                "over the full interaction",
                "over the visible interaction",
            ),
            instruction=condition.instruction.replace(
                "across the complete encounter",
                "across the visible interaction",
            ),
            output_type=condition.output_type,
        )
        for condition in MEDSP1000_JUDGING_PROFILE.conditions
    )
    return JudgingTaskProfile(
        conditions=conditions,
        user_prompt_builder=partial(
            _user_prompt,
            view_turn_count=view_turn_count,
        ),
    )


def build_medsp1000_judgment_keys(
    *,
    question_id: str,
    responses: Sequence[PocqiResponseInput],
    judge_model: str,
    settings: PocqiJudgingSettings,
    view_turn_count: int | None = None,
) -> dict[PocqiJudgingCase, str]:
    """Build stable MedSP1000 keys without making a model call."""

    return build_judgment_keys(
        question_id=question_id,
        responses=responses,
        judge_model=judge_model,
        settings=settings,
        profile=medsp1000_judging_profile(view_turn_count),
        judging_cases=MEDSP1000_JUDGING_CASES,
    )


def judge_medsp1000_trajectories(
    *,
    question_id: str,
    question_text: str,
    source_scenario_path: str,
    responses: Sequence[PocqiResponseInput],
    judge_model: str,
    settings: PocqiJudgingSettings,
    view_turn_count: int | None = None,
    model_caller: ModelCaller = call_model,
    output_paths: Mapping[PocqiJudgingCase, str | Path] | None = None,
    resume_tracker: PocqiResumeTracker | None = None,
) -> dict[PocqiJudgingCase, PocqiJudgmentRecord]:
    """Judge complete identity-blinded trajectories and append every attempt."""

    return judge_ranked_responses(
        question_id=question_id,
        question_text=question_text,
        specialty=source_scenario_path,
        responses=responses,
        judge_model=judge_model,
        settings=settings,
        profile=medsp1000_judging_profile(view_turn_count),
        model_caller=model_caller,
        output_paths=output_paths,
        resume_tracker=resume_tracker,
        judging_cases=MEDSP1000_JUDGING_CASES,
    )
