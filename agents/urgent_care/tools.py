"""Stub tools that stand in for physical bedside hardware.

The triage nurse and the radiologist normally rely on real devices: a
vitals monitor at the bedside and an imaging modality in the radiology
suite. In this simulation neither exists, so rather than asking the
patient to type numbers or upload files we expose stub tools that return
plausible canned data. In a future robotic deployment these stubs would
be replaced by drivers that talk to the actual hardware.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# Sample chest radiograph bundled with the repo. Used as the canned
# return value for `upload_chest_xray`.
_XRAY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "pleural_effusion.png"
)


def get_patient_vitals() -> dict[str, Any]:
    """Read the patient's current vital signs from the bedside monitor.

    Use this instead of asking the patient to type values. Returns blood
    pressure, heart rate, respiratory rate, temperature, oxygen saturation
    on room air, and pain score - the standard urgent-care vitals set.
    """
    # Numbers consistent with community-acquired pneumonia: febrile,
    # tachycardic, tachypneic, mildly hypoxemic, low-normal blood
    # pressure, pleuritic chest discomfort.
    return {
        "blood_pressure_mmHg": "108/68",
        "heart_rate_bpm": 112,
        "respiratory_rate_per_min": 26,
        "temperature_C": 38.9,
        "spo2_percent_room_air": 91,
        "pain_score_0_10": 4,
    }


def upload_chest_xray() -> dict[str, Any]:
    """Pull the patient's most recent chest radiograph from the imaging suite.

    Use this instead of asking the patient to upload a file. The image
    itself is surfaced into the conversation as an inline picture (via the
    `_image_path` key), so the radiologist can read it directly on the
    next turn.
    """
    return {
        "study": "PA chest radiograph",
        "_image_path": str(_XRAY_PATH),
    }
