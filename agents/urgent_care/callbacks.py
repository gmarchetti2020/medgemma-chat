"""Programmatic guards and context filters for the urgent-care agents.

Two kinds of helpers live here:

1. `gate_nurse_transfer` (an `after_model_callback`) prevents the nurse
   from transferring to the doctor before vitals are on record.

2. `doctor_handoff_filter` and `radiologist_handoff_filter` (both
   `before_model_callback`s) replace the verbose pre-handoff conversation
   history with a single structured message before the LLM call. The 4B
   model loses role identity over long, dense histories - the filters
   keep its input lean and on-topic, fixing a lot of the "doctor channels
   the nurse" / "radiologist hallucinates a tool response" degradation.
"""
from __future__ import annotations

import re
from typing import Optional

from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

# Each pattern must match somewhere in the nurse's final message for the
# transfer to be allowed. Patterns look for a label followed (within a short
# window) by a digit, which is a cheap proxy for "vital was actually
# recorded" rather than left as a placeholder.
_VITAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bBP\b[^A-Za-z\n]{0,40}\d", re.IGNORECASE),
    re.compile(r"(?:\bHR\b|heart\s*rate)[^A-Za-z\n]{0,40}\d", re.IGNORECASE),
    re.compile(
        r"(?:\bRR\b|resp(?:iratory)?\s*rate)[^A-Za-z\n]{0,40}\d", re.IGNORECASE
    ),
    re.compile(r"(?:\btemp(?:erature)?\b)[^A-Za-z\n]{0,40}\d", re.IGNORECASE),
    re.compile(
        r"(?:\bSpO2\b|\bO2\s*sat|\bsat(?:uration)?\b)[^A-Za-z\n]{0,40}\d",
        re.IGNORECASE,
    ),
    re.compile(r"\bpain\b[^A-Za-z\n]{0,40}\d", re.IGNORECASE),
]
_HEADER_RE = re.compile(r"triage\s+summary", re.IGNORECASE)
# Gemma3 emits thought-delimiter tokens like `<unused94>` / `<unused95>` that
# leak through `skip_special_tokens=True`. The thought block between them
# routinely contains the model echoing its own system prompt (with template
# placeholders like `<name>`, `<bpm>`, `<n>` etc.). We must strip the entire
# block before the placeholder scan, otherwise the gate sees template
# placeholders that aren't actually in the user-facing summary.
_GEMMA_THOUGHT_BLOCK_RE = re.compile(
    r"<\s*unused94\s*>.*?(?:<\s*unused95\s*>|$)",
    re.IGNORECASE | re.DOTALL,
)
_GEMMA_THOUGHT_TOK_RE = re.compile(r"<\s*unused\d+\s*>", re.IGNORECASE)
# A placeholder like "<...>" or "<n>" inside the supposed summary means the
# nurse hallucinated values it doesn't have.
_PLACEHOLDER_RE = re.compile(r"<\s*[^>]{0,30}\s*>")


def _join_text(response: LlmResponse) -> str:
    if not response.content or not response.content.parts:
        return ""
    return "\n".join(
        p.text for p in response.content.parts if getattr(p, "text", None)
    )


def _has_transfer_call(response: LlmResponse, target: str = "doctor") -> bool:
    if not response.content or not response.content.parts:
        return False
    for p in response.content.parts:
        fc = getattr(p, "function_call", None)
        if fc is not None and fc.name == "transfer_to_agent":
            args = dict(fc.args or {})
            if args.get("agent_name") == target:
                return True
    return False


def _triage_complete(text: str) -> bool:
    # Strip Gemma3 thought blocks (and any stray bare tokens) before checks -
    # the thought block is not user-facing and routinely echoes the system
    # prompt's template placeholders.
    scrub = _GEMMA_THOUGHT_BLOCK_RE.sub("", text)
    scrub = _GEMMA_THOUGHT_TOK_RE.sub("", scrub)
    if not _HEADER_RE.search(scrub):
        return False
    if _PLACEHOLDER_RE.search(scrub):
        return False
    return all(p.search(scrub) for p in _VITAL_PATTERNS)


def gate_nurse_transfer(callback_context, llm_response: LlmResponse) -> Optional[LlmResponse]:
    """Block premature nurse->doctor transfers.

    Returns a modified response when the nurse tried to transfer without a
    complete triage summary (vitals are entered manually by the patient, so
    the gate verifies the summary text carries real numeric values rather
    than placeholders). Returns None to leave the response untouched.
    """
    if not _has_transfer_call(llm_response, target="doctor"):
        return None

    text = _join_text(llm_response)
    if _triage_complete(text):
        return None

    # Drop only the doctor-targeted transfer call. Keep any text and any other
    # function calls (the nurse should not be issuing those, but we don't want
    # to be over-aggressive).
    kept_parts = []
    for p in llm_response.content.parts:
        fc = getattr(p, "function_call", None)
        if fc is not None and fc.name == "transfer_to_agent":
            args = dict(fc.args or {})
            if args.get("agent_name") == "doctor":
                continue
        kept_parts.append(p)
    if not kept_parts:
        kept_parts = [genai_types.Part(text="")]
    correction = (
        "\n\n(Triage is not yet complete. I still need to capture every "
        "required item before I can hand you over. Let's continue.)"
    )
    kept_parts.append(genai_types.Part(text=correction))

    return LlmResponse(
        content=genai_types.Content(role="model", parts=kept_parts),
        partial=False,
        turn_complete=True,
    )


# Minimum number of prior doctor dialogue turns required before the doctor is
# allowed to refer the patient to the radiologist. The 27B model otherwise
# jumps straight to imaging on its very first turn, before taking any focused
# history. Each unit is one doctor-authored, patient-facing message already on
# record, so a value of 2 forces roughly two history exchanges (e.g. HPI then
# PMH/social) before any imaging can be ordered.
_MIN_DOCTOR_HISTORY_TURNS = 2


def _count_author_text_turns(session, author: str) -> int:
    """Number of events authored by `author` that carry user-facing text."""
    n = 0
    for ev in getattr(session, "events", []) or []:
        if getattr(ev, "author", None) != author:
            continue
        c = getattr(ev, "content", None)
        if not c or not getattr(c, "parts", None):
            continue
        if any(getattr(p, "text", None) and p.text.strip() for p in c.parts):
            n += 1
    return n


def gate_doctor_referral(
    callback_context, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Block premature doctor->radiologist referrals.

    Mirrors `gate_nurse_transfer`: if the doctor tries to hand off to the
    radiologist before it has conducted at least `_MIN_DOCTOR_HISTORY_TURNS`
    of focused history-taking, strip the transfer call (keeping any dialogue
    text) and append a short nudge so the model takes a history first.
    Returns None to leave the response untouched.
    """
    if not _has_transfer_call(llm_response, target="radiologist"):
        return None

    session = getattr(callback_context, "session", None)
    if session is None:
        return None
    # Events from the response we're evaluating are not yet on the session, so
    # this counts strictly *prior* doctor turns.
    if _count_author_text_turns(session, "doctor") >= _MIN_DOCTOR_HISTORY_TURNS:
        return None

    kept_parts = []
    for p in llm_response.content.parts:
        fc = getattr(p, "function_call", None)
        if fc is not None and fc.name == "transfer_to_agent":
            args = dict(fc.args or {})
            if args.get("agent_name") == "radiologist":
                continue
        kept_parts.append(p)
    if not kept_parts:
        kept_parts = [genai_types.Part(text="")]
    correction = (
        "\n\n(Before I order any imaging I should take a proper focused "
        "history. Let me continue questioning the patient - HPI details, past "
        "medical history, and relevant social history - before referring to "
        "radiology.)"
    )
    kept_parts.append(genai_types.Part(text=correction))

    return LlmResponse(
        content=genai_types.Content(role="model", parts=kept_parts),
        partial=False,
        turn_complete=True,
    )


# ---------------------------------------------------------------------------
# Handoff filters (before_model_callback)
# ---------------------------------------------------------------------------


def _collect_user_dialogue(events) -> list[str]:
    """All user-authored text from `events`, in order."""
    out: list[str] = []
    for ev in events:
        if getattr(ev, "author", None) != "user":
            continue
        c = getattr(ev, "content", None)
        if not c or not getattr(c, "parts", None):
            continue
        for p in c.parts:
            if getattr(p, "text", None):
                out.append(p.text.strip())
    return out


def _last_text_from_author(events, author: str) -> Optional[str]:
    """Concatenated text of the most recent event authored by `author`."""
    last: Optional[str] = None
    for ev in events:
        if getattr(ev, "author", None) != author:
            continue
        c = getattr(ev, "content", None)
        if not c or not getattr(c, "parts", None):
            continue
        chunks = [p.text for p in c.parts if getattr(p, "text", None)]
        if chunks:
            last = "\n".join(chunks).strip()
    return last


def _build_nurse_to_doctor_handoff(pre_events) -> str:
    """Compact summary of the nurse's session for the doctor's input."""
    user_dialogue = _collect_user_dialogue(pre_events)
    lines = [
        "[Nurse handoff to physician]",
        "",
        "Patient statements (verbatim, in chronological order). The patient's "
        "manually entered vital signs are among these statements:",
    ]
    if user_dialogue:
        for utterance in user_dialogue:
            lines.append(f"- {utterance}")
    else:
        lines.append("- (no patient statements on record)")
    lines.append("")
    lines.append(
        "You are the urgent-care physician. Acknowledge the patient by "
        "name, take a focused HPI/PMH, build a brief differential, refer "
        "to the radiologist if imaging will change management, and close "
        "with an ASSESSMENT AND PLAN. Do NOT re-introduce yourself as the "
        "nurse and do NOT reproduce the TRIAGE SUMMARY."
    )
    return "\n".join(lines)


def _build_doctor_to_radiologist_handoff(pre_events) -> str:
    """Compact referral note for the radiologist's input.

    We drop the doctor's earlier deliberation but keep its referral message,
    which states the ordered study (the modality may be any X-ray, ultrasound,
    CT, or MRI) and the clinical question. That, plus the patient's verbatim
    statements, is everything the radiologist needs.
    """
    referral = _last_text_from_author(pre_events, "doctor")

    lines = [
        "[Physician referral to radiologist]",
        "",
    ]
    if referral:
        lines.append(
            "Referring physician's note (states the study requested - the "
            "modality and body region - and the clinical question):"
        )
        lines.append(referral)
        lines.append("")
    else:
        lines.append("The urgent-care physician has referred this patient for imaging.")
        lines.append("")
    lines.append(
        "You are the radiologist. Identify the study the physician ordered "
        "(any X-ray, ultrasound, CT, or MRI). Ask the patient to upload that "
        "study "
        "with the upload_imaging_study tool, read the image once it appears in "
        "the next turn, write a structured RADIOLOGY REPORT in plain text (NOT "
        "JSON) appropriate to the modality, and transfer back to the doctor."
    )
    return "\n".join(lines)


def _filter_history_with_handoff(
    callback_context, llm_request: LlmRequest, target_author: str, build_handoff
) -> None:
    """Replace pre-handoff events with a synthesized handoff message.

    Everything before `target_author`'s first event is collapsed into one
    `user`-role message produced by `build_handoff(pre_events)`. Events
    from `target_author`'s first turn onward are kept as-is so its own
    tool-call/response chains remain intact.
    """
    session = getattr(callback_context, "session", None)
    if session is None:
        return
    events = list(getattr(session, "events", []) or [])
    first_idx = next(
        (
            i
            for i, ev in enumerate(events)
            if getattr(ev, "author", None) == target_author
        ),
        len(events),
    )
    pre, post = events[:first_idx], events[first_idx:]
    handoff_text = build_handoff(pre)

    new_contents: list[genai_types.Content] = [
        genai_types.Content(
            role="user", parts=[genai_types.Part(text=handoff_text)]
        )
    ]
    for ev in post:
        c = getattr(ev, "content", None)
        if c is not None:
            new_contents.append(c)
    llm_request.contents = new_contents


def doctor_handoff_filter(
    callback_context, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Strip nurse-side noise from the doctor's input on every doctor turn."""
    _filter_history_with_handoff(
        callback_context,
        llm_request,
        target_author="doctor",
        build_handoff=_build_nurse_to_doctor_handoff,
    )
    return None


def radiologist_handoff_filter(
    callback_context, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Strip nurse + doctor deliberation from the radiologist's input."""
    _filter_history_with_handoff(
        callback_context,
        llm_request,
        target_author="radiologist",
        build_handoff=_build_doctor_to_radiologist_handoff,
    )
    return None
