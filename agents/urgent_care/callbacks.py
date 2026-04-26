"""Programmatic guards for agents that supplement prompt-only rules.

The nurse must not transfer to the doctor until a complete triage summary
(with numeric vitals) has actually been written. A 4B model can be coaxed by
prompts but not reliably constrained, so we enforce it here in code: if the
nurse emits a `transfer_to_agent` call without a completed TRIAGE SUMMARY in
the same response, we strip the call and append a self-correction.
"""
from __future__ import annotations

import re
from typing import Optional

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
# A placeholder like "<...>" or "<n>" inside the supposed summary means the
# nurse hallucinated values it doesn't have.
_PLACEHOLDER_RE = re.compile(r"<\s*[^>]{0,30}\s*>")


def _join_text(response: LlmResponse) -> str:
    if not response.content or not response.content.parts:
        return ""
    return "\n".join(
        p.text for p in response.content.parts if getattr(p, "text", None)
    )


def _has_transfer_call(response: LlmResponse) -> bool:
    if not response.content or not response.content.parts:
        return False
    for p in response.content.parts:
        fc = getattr(p, "function_call", None)
        if fc is not None and fc.name == "transfer_to_agent":
            return True
    return False


def _triage_complete(text: str) -> bool:
    if not _HEADER_RE.search(text):
        return False
    if _PLACEHOLDER_RE.search(text):
        return False
    return all(p.search(text) for p in _VITAL_PATTERNS)


def gate_nurse_transfer(callback_context, llm_response: LlmResponse) -> Optional[LlmResponse]:
    """Block premature nurse->doctor transfers.

    Returns a modified response when the nurse tried to transfer without a
    complete triage summary. Returns None to leave the response untouched.
    """
    if not _has_transfer_call(llm_response):
        return None

    text = _join_text(llm_response)
    if _triage_complete(text):
        return None

    # Drop the function_call, keep any text the nurse produced, and append a
    # self-correction so the next user turn naturally continues the triage.
    kept_parts = []
    for p in llm_response.content.parts:
        fc = getattr(p, "function_call", None)
        if fc is not None and fc.name == "transfer_to_agent":
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
