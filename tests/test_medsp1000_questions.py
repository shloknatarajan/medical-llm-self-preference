from pathlib import Path

from scripts.prepare_medsp1000_questions import (
    blocking_reason,
    classify_scenario,
    select_role_files,
)


def test_blocking_reason_detects_invalid_patient_roles() -> None:
    assert (
        blocking_reason("You are the mother of a sick child.", "parent.md")
        == "proxy_or_caregiver"
    )
    assert (
        blocking_reason("You are Jane, the mother of a toddler.", "case.md")
        == "proxy_or_caregiver"
    )
    assert (
        blocking_reason("The newborn is here for follow-up.", "case.md")
        == "pediatric_proxy_or_preverbal"
    )
    assert (
        blocking_reason("The patient is currently intubated.", "case.md")
        == "nonresponsive_patient"
    )
    assert blocking_reason("Correct diagnosis: pneumonia", "case.md") == "diagnosis_key_leakage"
    assert blocking_reason("Diagnosis: pneumonia", "case.md") == "diagnosis_key_leakage"
    assert blocking_reason("You have had a cough for two weeks.", "SP script.md") is None


def test_patient_file_selection_avoids_checklist(tmp_path: Path) -> None:
    patient = tmp_path / "Standardized Patient Case.md"
    checklist = tmp_path / "SP Checklist.md"
    patient.write_text("patient history", encoding="utf-8")
    checklist.write_text("checklist" * 100, encoding="utf-8")
    selected = select_role_files(
        tmp_path,
        [patient.name, checklist.name],
        "patient",
    )
    assert selected == (patient,)


def test_screen_accepts_detailed_interactive_patient() -> None:
    actor = """
    Instructions for Patient: You have experienced a cough for two months.
    Chief complaint: cough. Past medical history: none. Medications: none.
    Allergies: penicillin. Social history: smokes daily. Family history: cancer.
    If asked by the doctor, explain that you tried to quit. Do not volunteer this.
    """ * 8
    assert classify_scenario(actor)["strict_core_candidate"] is True


def test_screen_rejects_parent_proxy() -> None:
    actor = """
    Standardized patient role. You are the mother of a sick child. Chief complaint:
    fever. Past medical history: asthma. Medications: albuterol. Allergies: none.
    If asked by the physician, describe the child's symptoms.
    """ * 8
    assert classify_scenario(actor)["strict_core_candidate"] is False
