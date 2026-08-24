"""Build the deterministic MedSP1000 multi-turn question set."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASET_ID = "byrLLCC/MedSP1000"
DATASET_REVISION = "55e3e55efd08c73baab912ba0c5b42637114fbc8"
SCHEMA_VERSION = "2.0"
QUESTION_TYPE = "multi_turn_standardized_patient"
DEFAULT_SEED = 42

HISTORY_DOMAINS: dict[str, tuple[str, ...]] = {
    "present_illness": (
        "history of present illness",
        "chief complaint",
        "presenting complaint",
        "current symptoms",
        "symptoms",
    ),
    "past_history": ("past medical history", "medical history", "past history"),
    "medications": ("medications", "current medication", "medicines"),
    "allergies": ("allergies", "allergy"),
    "social_history": ("social history", "smoking history", "substance use"),
    "family_history": ("family history",),
    "review_of_systems": ("review of systems", "pertinent negatives"),
    "actor_behavior": (
        "if asked",
        "when asked",
        "only if asked",
        "if the doctor",
        "if the student",
        "opening statement",
        "affect",
        "demeanor",
    ),
}

PATIENT_SIGNALS = (
    "standardized patient",
    "simulated patient",
    "instructions for patient",
    "patient instructions",
    "patient profile",
    "patient role",
    "you are a patient",
    "you have experienced",
    "you have been",
)

INTERACTION_SIGNALS = (
    "the doctor",
    "the physician",
    "the student",
    "the clinician",
    "the resident",
    "if asked",
    "when asked",
    "tell them",
    "do not volunteer",
)

PROXY_PATTERNS = (
    re.compile(
        r"\byou are (?:the |his |her )?"
        r"(?:mother|father|parent|wife|husband|daughter|son|caregiver)\b"
    ),
    re.compile(
        r"\brole\s*:\s*(?:mother|father|parent|wife|husband|daughter|son|caregiver)\b"
    ),
    re.compile(r"\bstandardized (?:parent|caregiver|family member)\b"),
)

NONPATIENT_ROLE_PATTERNS = (
    re.compile(
        r"\byou are (?:an? |the )?(?:nurse|physician|doctor|surgeon|resident|medical student)\b"
    ),
    re.compile(
        r"\brole\s*:\s*(?:nurse|physician|doctor|surgeon|resident|medical student)\b"
    ),
)

AUXILIARY_FILE_TERMS = (
    "checklist",
    "evaluation",
    "evaluator",
    "rating",
    "questionnaire",
    "clinical pictures",
    "documentation exercise",
)
PATIENT_FILE_POSITIVE_TERMS = (
    "sp ",
    "sp_",
    "standardized patient",
    "simulated patient",
    "patient script",
    "patient case",
    "patient profile",
    "patient information",
    "patient instructions",
    "training materials",
)
CLINICIAN_FILE_POSITIVE_TERMS = (
    "examinee",
    "student instructions",
    "pre-encounter",
    "preencounter",
    "door note",
    "clipboard",
    "case",
)

BLOCKING_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "proxy_or_caregiver": (
        re.compile(
            r"\byou are (?:the |his |her |a )?"
            r"(?:mother|father|parent|caregiver|spouse|wife|husband|daughter|son)\b",
            re.I,
        ),
        re.compile(r"\byou are [^\n.]{0,50}\b(?:mother|father|parents?|caregiver) of\b", re.I),
        re.compile(r"\byou are the parents? of\b", re.I),
        re.compile(r"\byou are the sole caregiver for\b", re.I),
        re.compile(
            r"\brole\s*:\s*"
            r"(?:mother|father|parent|caregiver|spouse|wife|husband|daughter|son)\b",
            re.I,
        ),
        re.compile(r"\bstandardized (?:parent|caregiver|family member)\b", re.I),
        re.compile(r"\bparent/child scenario\b", re.I),
    ),
    "pediatric_proxy_or_preverbal": (
        re.compile(r"\b(?:newborn|infant|baby)\b", re.I),
    ),
    "nonresponsive_patient": (
        re.compile(
            r"\b(?:patient is |you are )(?:currently )?"
            r"(?:unresponsive|unconscious|intubated|sedated)\b",
            re.I,
        ),
        re.compile(r"\b(?:cardiac arrest|death by neurologic criteria|brain dead)\b", re.I),
    ),
    "nonpatient_actor": (
        re.compile(
            r"\byou are (?:an? |the )?"
            r"(?:nurse|physician|doctor|surgeon|resident|medical student)\b",
            re.I,
        ),
        re.compile(
            r"\brole\s*:\s*"
            r"(?:nurse|physician|doctor|surgeon|resident|medical student)\b",
            re.I,
        ),
    ),
    "diagnosis_key_leakage": (
        re.compile(r"\b(?:correct|final) diagnosis\s*:", re.I),
        re.compile(r"(?m)^\s*\|?\s*(?:actual\s+)?diagnosis\s*:", re.I),
        re.compile(r"\banswer key\b", re.I),
    ),
    "non_english_named_case": (
        re.compile(r"\b(?:spanish|tagalog|igbo|french|chinese|mandarin)-language\b", re.I),
        re.compile(r"\bsp case - (?:spanish|tagalog|igbo|french|chinese)\b", re.I),
    ),
}


@dataclass(frozen=True, slots=True)
class Candidate:
    scenario: dict[str, Any]
    actor_files: tuple[Path, ...]
    examinee_files: tuple[Path, ...]
    actor_material: str
    examinee_material: str
    official_q100: bool
    quality_score: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--source", type=Path, default=Path("data/source/medsp1000_repo")
    )
    parser.add_argument(
        "--official-subset",
        type=Path,
        default=Path("data/source/medsp1000_code/subset.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/question_sets/medsp1000_generation_cases.jsonl"),
    )
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def patient_file_score(path: Path) -> tuple[int, int, str]:
    name = path.name.lower()
    score = sum(term in name for term in PATIENT_FILE_POSITIVE_TERMS) * 3
    score -= sum(term in name for term in AUXILIARY_FILE_TERMS) * 6
    return score, path.stat().st_size, path.as_posix()


def clinician_file_score(path: Path) -> tuple[int, int, str]:
    name = path.name.lower()
    score = sum(term in name for term in CLINICIAN_FILE_POSITIVE_TERMS) * 3
    score -= sum(term in name for term in AUXILIARY_FILE_TERMS) * 5
    return score, -path.stat().st_size, path.as_posix()


def select_role_files(
    source: Path, relative_paths: list[str], role: str
) -> tuple[Path, ...]:
    files = [
        source / relative
        for relative in relative_paths
        if "-zh" not in Path(relative).stem.lower()
    ]
    if not files:
        return ()
    if role == "patient":
        non_auxiliary = [
            path
            for path in files
            if not any(term in path.name.lower() for term in AUXILIARY_FILE_TERMS)
        ]
        pool = non_auxiliary or files
        chosen = max(pool, key=patient_file_score)
        return (chosen,)
    if role == "clinician":
        pool = [
            path
            for path in files
            if not any(term in path.name.lower() for term in AUXILIARY_FILE_TERMS)
        ]
        if not pool:
            return ()
        total_characters = sum(len(read_text(path)) for path in pool)
        if total_characters <= 16_000:
            return tuple(sorted(pool))
        chosen = max(pool, key=clinician_file_score)
        return (chosen,)
    raise ValueError(f"unknown role: {role}")


def combine_files(files: tuple[Path, ...]) -> str:
    return "\n\n".join(read_text(path) for path in files).strip()


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def classify_scenario(actor_text: str) -> dict[str, Any]:
    """Return the conservative screen used to select interactive patient cases."""

    actor_lower = actor_text.lower()
    history_domains = sorted(
        name
        for name, terms in HISTORY_DOMAINS.items()
        if contains_any(actor_lower, terms)
    )
    patient_signal = contains_any(actor_lower, PATIENT_SIGNALS)
    interaction_signal = contains_any(actor_lower, INTERACTION_SIGNALS)
    proxy_signal = any(pattern.search(actor_lower) for pattern in PROXY_PATTERNS)
    nonpatient_role_signal = any(
        pattern.search(actor_lower) for pattern in NONPATIENT_ROLE_PATTERNS
    )
    strict_core_candidate = (
        patient_signal
        and interaction_signal
        and len(history_domains) >= 4
        and len(actor_text) >= 2_000
        and not proxy_signal
        and not nonpatient_role_signal
    )
    return {
        "strict_core_candidate": strict_core_candidate,
        "history_domains": history_domains,
    }


def scan_scenarios(source: Path) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    scenario_dirs = sorted(
        role_dir.parent
        for role_dir in source.glob("*/scenario*/sp_actor")
        if role_dir.is_dir()
    )
    for scenario_dir in scenario_dirs:
        actor_files = sorted(
            path
            for path in (scenario_dir / "sp_actor").rglob("*")
            if path.is_file() and "-zh" not in path.stem.lower()
        )
        examinee_files = sorted(
            path
            for path in (scenario_dir / "examinee").rglob("*")
            if path.is_file() and "-zh" not in path.stem.lower()
        )
        actor_text = "\n\n".join(read_text(path) for path in actor_files)
        scenario_path = scenario_dir.relative_to(source).as_posix()
        scenarios.append(
            {
                "scenario_path": scenario_path,
                "actor_files": [path.relative_to(source).as_posix() for path in actor_files],
                "examinee_files": [
                    path.relative_to(source).as_posix() for path in examinee_files
                ],
                **classify_scenario(actor_text),
            }
        )
    return scenarios


def blocking_reason(actor_material: str, actor_filenames: str) -> str | None:
    searchable = f"{actor_filenames}\n{actor_material}"
    for reason, patterns in BLOCKING_PATTERNS.items():
        if any(pattern.search(searchable) for pattern in patterns):
            return reason
    return None


def load_official_subset(path: Path) -> tuple[set[str], str | None]:
    if not path.exists():
        return set(), None
    raw = json.loads(path.read_text(encoding="utf-8"))
    code_root = path.parent
    try:
        revision = subprocess.run(
            ["git", "-C", str(code_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return set(map(str, raw["scenarios"])), revision


def verify_source_revision(source: Path) -> None:
    """Refuse to label source files with a revision they did not come from."""
    try:
        revision = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"cannot verify MedSP1000 source revision at {source}") from exc
    if revision != DATASET_REVISION:
        raise ValueError(
            f"MedSP1000 source revision is {revision}, expected {DATASET_REVISION}"
        )
    if changes:
        raise ValueError(f"MedSP1000 source repository has uncommitted changes: {source}")


def build_candidates(args: argparse.Namespace) -> tuple[list[Candidate], Counter[str], str | None]:
    official_subset, official_code_revision = load_official_subset(args.official_subset)
    rejection_counts: Counter[str] = Counter()
    candidates: list[Candidate] = []
    for scenario in scan_scenarios(args.source):
        if not scenario.get("strict_core_candidate"):
            rejection_counts["not_strict_core"] += 1
            continue
        actor_files = select_role_files(args.source, scenario["actor_files"], "patient")
        examinee_files = select_role_files(
            args.source, scenario["examinee_files"], "clinician"
        )
        if not actor_files or not examinee_files:
            rejection_counts["missing_role_material"] += 1
            continue
        actor_material = combine_files(actor_files)
        examinee_material = combine_files(examinee_files)
        if not 2_000 <= len(actor_material) <= 24_000:
            rejection_counts["patient_material_length"] += 1
            continue
        if not 50 <= len(examinee_material) <= 24_000:
            rejection_counts["clinician_material_length"] += 1
            continue
        reason = blocking_reason(actor_material, "\n".join(path.name for path in actor_files))
        if reason:
            rejection_counts[reason] += 1
            continue
        quality_score = (
            int(bool(scenario.get("strict_core_candidate"))) * 5
            + min(len(scenario.get("history_domains", [])), 6)
            + int(len(scenario.get("actor_files", [])) == 1) * 2
            + int(len(scenario.get("examinee_files", [])) == 1)
            + int(scenario["scenario_path"] in official_subset) * 2
        )
        candidates.append(
            Candidate(
                scenario=scenario,
                actor_files=actor_files,
                examinee_files=examinee_files,
                actor_material=actor_material,
                examinee_material=examinee_material,
                official_q100=scenario["scenario_path"] in official_subset,
                quality_score=quality_score,
            )
        )
    return candidates, rejection_counts, official_code_revision


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    verify_source_revision(args.source)
    candidates, rejection_counts, code_revision = build_candidates(args)

    # Exact duplicate patient cards do not provide independent scenarios.
    unique: list[Candidate] = []
    seen_actor_hashes: set[str] = set()
    for candidate in sorted(
        candidates, key=lambda item: item.scenario["scenario_path"]
    ):
        actor_hash = sha256_text(candidate.actor_material)
        if actor_hash in seen_actor_hashes:
            rejection_counts["duplicate_patient_material"] += 1
            continue
        seen_actor_hashes.add(actor_hash)
        unique.append(candidate)
    if len(unique) < args.count:
        raise ValueError(
            f"only {len(unique)} eligible unique scenarios remain; requested {args.count}; "
            f"rejections={dict(rejection_counts)}"
        )

    # Shuffle within quality bands so the cohort is deterministic without simply
    # taking the earliest MedEdPORTAL identifiers.
    rng = random.Random(args.seed)
    by_score: dict[int, list[Candidate]] = {}
    for candidate in unique:
        by_score.setdefault(candidate.quality_score, []).append(candidate)
    ranked: list[Candidate] = []
    for score in sorted(by_score, reverse=True):
        group = by_score[score]
        rng.shuffle(group)
        ranked.extend(group)
    selected = ranked[: args.count]

    records: list[dict[str, Any]] = []
    for cohort_index, candidate in enumerate(selected, start=1):
        scenario = candidate.scenario
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "question_id": scenario["scenario_path"].replace("/", "__"),
                "question_type": QUESTION_TYPE,
                "question_text": candidate.examinee_material,
                "private_patient_context": candidate.actor_material,
                "source_dataset": DATASET_ID,
                "source_revision": DATASET_REVISION,
                "source_scenario_path": scenario["scenario_path"],
                "cohort_index": cohort_index,
                "selection_reason": "strict interactive patient generation cohort",
                "selection_seed": args.seed,
                "quality_score": candidate.quality_score,
                "official_q100_member": candidate.official_q100,
                "history_domains": scenario["history_domains"],
                "private_patient_context_sha256": sha256_text(candidate.actor_material),
                "question_text_sha256": sha256_text(candidate.examinee_material),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    args.output.write_text(artifact, encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact": args.output.name,
        "artifact_sha256": hashlib.sha256(artifact.encode("utf-8")).hexdigest(),
        "record_count": len(records),
        "question_type": QUESTION_TYPE,
        "source_dataset": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "official_code_revision": code_revision,
        "selection_seed": args.seed,
        "eligible_unique_scenarios": len(unique),
        "official_q100_members_selected": sum(item.official_q100 for item in selected),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "included_roles": ["sp_actor", "examinee"],
        "excluded_roles": ["environment_controller", "evaluator"],
        "generation_only": True,
        "grading_or_judging": False,
        "record_schema": {
            "schema_version": "string",
            "question_id": "string (deterministic from source_scenario_path)",
            "question_type": f"constant: {QUESTION_TYPE}",
            "question_text": "string (private clinician-visible initialization)",
            "private_patient_context": "string (private patient-model context)",
            "source_dataset": "string",
            "source_revision": "string",
            "source_scenario_path": "string",
            "question_text_sha256": "SHA-256 hex string",
            "private_patient_context_sha256": "SHA-256 hex string",
            "cohort_index": "integer",
            "selection_reason": "string",
            "selection_seed": "integer",
            "quality_score": "integer",
            "official_q100_member": "boolean",
            "history_domains": "array[string]",
        },
        "manual_review_note": (
            "The five-case pilot was manually inspected. The expanded cohort passed "
            "deterministic structural and leakage screens but is not a substitute for "
            "manual review of every source packet."
        ),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
