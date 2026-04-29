"""Urgent-care multi-agent app: nurse -> doctor -> radiologist.

Run with `adk web` from the repo root:

    adk web agents

Then open the printed URL and pick `urgent_care` from the agent dropdown.
"""
from __future__ import annotations

from google.adk.agents import LlmAgent

from .callbacks import gate_nurse_transfer
from .medgemma_llm import MedGemmaLlm
from .prompts import (
    DOCTOR_INSTRUCTION,
    NURSE_INSTRUCTION,
    RADIOLOGIST_INSTRUCTION,
)

# A single MedGemmaLlm instance is shared across all three agents. The model
# weights themselves live in a process-wide singleton (see medgemma_llm.py),
# so we never load the 4B model more than once.
_llm = MedGemmaLlm()


radiologist_agent = LlmAgent(
    name="radiologist",
    description=(
        "Radiologist who reviews uploaded chest X-rays, abdominal films, or "
        "CT slices and produces a structured radiology report for the "
        "referring physician."
    ),
    model=_llm,
    instruction=RADIOLOGIST_INSTRUCTION,
)


doctor_agent = LlmAgent(
    name="doctor",
    description=(
        "Urgent-care physician who reviews the nurse's triage notes, takes a "
        "focused history, optionally refers to the radiologist for imaging, "
        "and finally produces an assessment and treatment plan."
    ),
    model=_llm,
    instruction=DOCTOR_INSTRUCTION,
    sub_agents=[radiologist_agent],
    # NOTE: We deliberately do NOT set `disallow_transfer_to_parent=True` on
    # the doctor. ADK's runner uses that same flag to decide whether an agent
    # can be picked as the active agent across user turns
    # (`_is_transferable_across_agent_tree`); setting it would force ADK to
    # fall back to the root (nurse) on every turn after the hand-off, and the
    # doctor + radiologist would never get a turn of their own. We enforce
    # "triage never reverses" via the doctor's prompt instead.
)


nurse_agent = LlmAgent(
    name="nurse",
    description=(
        "Triage nurse who collects chief complaint, key symptoms, and vital "
        "signs (entered manually by the patient) before handing off to the "
        "physician."
    ),
    model=_llm,
    instruction=NURSE_INSTRUCTION,
    sub_agents=[doctor_agent],
    # Hard guard: if the model tries to transfer before all required triage
    # items (including numeric vitals) are present in the message, strip the
    # transfer call and force it to keep asking.
    after_model_callback=gate_nurse_transfer,
)


# `adk web` looks for `agent.root_agent`.
root_agent = nurse_agent
