"""Tools that solicit real input from the patient.

The triage nurse and the radiologist normally rely on real devices: a
vitals monitor at the bedside and an imaging modality in the radiology
suite. Neither is connected in this deployment, so instead of fabricating
data these tools ask the patient to supply it directly - the nurse's tool
prompts the patient to type their vital signs, and the radiologist's tool
prompts the patient to upload their chest radiograph. The agent relays the
request, the patient responds on the next turn, and the agent reads the
values / image out of the conversation. In a future robotic deployment
these tools would instead talk to the actual hardware.
"""
from __future__ import annotations

from typing import Any


def get_patient_vitals() -> dict[str, Any]:
    """Request the patient's vital signs for manual entry.

    The bedside monitor is not connected, so this tool does NOT return
    readings. It returns instructions telling you to ask the patient to read
    each value off their home devices (BP cuff, thermometer, pulse oximeter)
    and type it in. Record the numbers the patient provides; never invent
    them.
    """
    return {
        "status": "manual_entry_required",
        "instructions": (
            "The bedside monitor is offline. Ask the patient to provide each "
            "of the following and type the values back to you: blood pressure "
            "(systolic/diastolic, mmHg), heart rate (bpm), respiratory rate "
            "(breaths/min), temperature (°C), oxygen saturation (% on room "
            "air), and pain score (0-10). If the patient cannot measure a "
            "value, record 'not available'."
        ),
        "required_fields": [
            "blood_pressure_mmHg",
            "heart_rate_bpm",
            "respiratory_rate_per_min",
            "temperature_C",
            "spo2_percent_room_air",
            "pain_score_0_10",
        ],
    }


def upload_chest_xray() -> dict[str, Any]:
    """Request the patient to upload their chest radiograph.

    No imaging modality is wired into this room, so this tool does NOT return
    an image. It returns instructions telling you to ask the patient to
    upload their most recent chest X-ray (or other requested study) as an
    image attachment. Once the patient attaches it, the image appears inline
    in the conversation and you can read it directly on the next turn.
    """
    return {
        "status": "upload_required",
        "instructions": (
            "No imaging modality is connected. Ask the patient to upload "
            "their most recent chest X-ray using the image / attachment "
            "button. Once the image appears in the conversation, read it and "
            "write your report. If the patient provides no image, say you "
            "cannot report without one."
        ),
    }
