"""Custom ADK BaseLlm wrapper around google/medgemma-27b-it.

Loads the model once per process (singleton) and shares it across all
agents in the urgent-care app. Translates ADK's LlmRequest into the
gemma3 chat-template format (text + images), runs generation, and
parses explicit `[[TRANSFER:agent_name]]` and `[[TOOL:tool_name]]`
markers out of the model output so we can hand off control and invoke
tools via ADK's normal flow even though MedGemma cannot natively emit
structured function calls. Override the default model with the
`MEDGEMMA_MODEL_ID` env var (e.g. `google/medgemma-1.5-4b-it` for a
faster, lower-quality run).
"""
from __future__ import annotations

import io
import os
import re
import threading
from typing import AsyncGenerator, Optional

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types
from PIL import Image

MODEL_ID = os.environ.get("MEDGEMMA_MODEL_ID", "google/medgemma-27b-it")
# Markers are case-insensitive AND tolerate single OR double brackets: the
# 4B model lowercases the keyword and frequently drops the second bracket.
TRANSFER_RE = re.compile(
    r"\[\[?\s*TRANSFER\s*:\s*([A-Za-z0-9_\-]+)\s*\]\]?", re.IGNORECASE
)
TOOL_RE = re.compile(
    r"\[\[?\s*TOOL\s*:\s*([A-Za-z0-9_\-]+)\s*\]\]?", re.IGNORECASE
)
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MEDGEMMA_MAX_NEW_TOKENS", "3072"))
DEFAULT_TEMPERATURE = float(os.environ.get("MEDGEMMA_TEMPERATURE", "0.7"))
DEFAULT_REPETITION_PENALTY = float(
    os.environ.get("MEDGEMMA_REPETITION_PENALTY", "1.1")
)


class _ModelHolder:
    """Process-wide lazy singleton so the model weights are loaded once."""

    _lock = threading.Lock()
    _model = None
    _processor = None

    @classmethod
    def get(cls):
        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    import torch
                    from transformers import (
                        AutoModelForImageTextToText,
                        AutoProcessor,
                    )

                    if torch.cuda.is_available():
                        dtype = torch.bfloat16
                        device_map = "cuda:0"
                    elif (
                        hasattr(torch, "backends")
                        and hasattr(torch.backends, "mps")
                        and torch.backends.mps.is_available()
                    ):
                        dtype = torch.float16
                        device_map = "mps"
                    else:
                        dtype = torch.float32
                        device_map = None

                    cls._processor = AutoProcessor.from_pretrained(MODEL_ID)
                    cls._model = AutoModelForImageTextToText.from_pretrained(
                        MODEL_ID,
                        dtype=dtype,
                        device_map=device_map,
                    )
                    cls._model.eval()
        return cls._model, cls._processor


def _system_text(llm_request: LlmRequest) -> Optional[str]:
    cfg = llm_request.config
    if cfg is None:
        return None
    si = getattr(cfg, "system_instruction", None)
    if si is None:
        return None
    if isinstance(si, str):
        return si
    parts = getattr(si, "parts", None)
    if parts:
        chunks = [p.text for p in parts if getattr(p, "text", None)]
        if chunks:
            return "\n".join(chunks)
    return str(si)


def _content_to_messages(
    contents: list[genai_types.Content], system_text: Optional[str]
) -> list[dict]:
    """Convert ADK Content list to gemma3 chat-template messages.

    Gemma3's chat template strictly requires user/assistant alternation and a
    leading user turn. ADK normally produces this, but transfers add a
    function_response turn that can land next to a user turn, and tool-only
    turns can collapse to empty content. We normalize all of that here.
    """
    raw: list[dict] = []
    for c in contents or []:
        role = "assistant" if c.role == "model" else "user"
        out_content: list[dict] = []
        for p in c.parts or []:
            if getattr(p, "text", None):
                out_content.append({"type": "text", "text": p.text})
            elif getattr(p, "inline_data", None) is not None:
                mime = (p.inline_data.mime_type or "").lower()
                if mime.startswith("image/") and p.inline_data.data:
                    img = Image.open(io.BytesIO(p.inline_data.data)).convert("RGB")
                    out_content.append({"type": "image", "image": img})
            elif getattr(p, "function_call", None) is not None:
                fc = p.function_call
                out_content.append(
                    {
                        "type": "text",
                        "text": (
                            f"[I called tool `{fc.name}` with args "
                            f"{dict(fc.args or {})}]"
                        ),
                    }
                )
            elif getattr(p, "function_response", None) is not None:
                fr = p.function_response
                # Tool stubs that produce an image (e.g. `upload_chest_xray`)
                # smuggle the local file path back via an `_image_path` key
                # in the response dict. Pop it out, render the rest as text
                # for the transcript, and inline the image so the model can
                # actually see it on the next turn.
                response = fr.response
                image_path: Optional[str] = None
                if isinstance(response, dict) and "_image_path" in response:
                    response = dict(response)
                    image_path = response.pop("_image_path", None)
                out_content.append(
                    {
                        "type": "text",
                        "text": (
                            f"[Tool `{fr.name}` returned: {response}]"
                        ),
                    }
                )
                if image_path:
                    try:
                        img = Image.open(image_path).convert("RGB")
                        out_content.append({"type": "image", "image": img})
                    except (OSError, ValueError):
                        out_content.append(
                            {
                                "type": "text",
                                "text": (
                                    f"[Tool `{fr.name}` reported image at "
                                    f"{image_path} but it could not be loaded.]"
                                ),
                            }
                        )
        if out_content:
            raw.append({"role": role, "content": out_content})

    # Merge consecutive same-role messages so user/assistant strictly alternate.
    merged: list[dict] = []
    for m in raw:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] = merged[-1]["content"] + m["content"]
        else:
            merged.append(m)

    # Inject the full system text at the head of the FIRST user turn
    # (gemma3's template does not consistently accept a `system` role).
    if system_text:
        injected = {"type": "text", "text": system_text}
        for m in merged:
            if m["role"] == "user":
                m["content"] = [injected] + m["content"]
                break
        else:
            merged.insert(0, {"role": "user", "content": [injected]})

    # Gemma3 requires the conversation to start with a user turn.
    if merged and merged[0]["role"] != "user":
        merged.insert(
            0,
            {
                "role": "user",
                "content": [{"type": "text", "text": "(begin consultation)"}],
            },
        )
    return merged


def _split_markers(text: str) -> tuple[str, list[str], Optional[str]]:
    """Pull `[[TOOL:name]]` and `[[TRANSFER:agent]]` markers out of model output.

    Returns the cleaned user-facing text, the list of tool names the model
    asked to invoke (in order), and the transfer target if any. The model
    only ever calls parameter-less tools, so we don't parse args.
    """
    # Dedupe while preserving order - the 4B model sometimes spams the same
    # marker several times in a single response.
    raw = [m.strip() for m in TOOL_RE.findall(text)]
    tool_names = list(dict.fromkeys(raw))
    cleaned = TOOL_RE.sub("", text)
    m = TRANSFER_RE.search(cleaned)
    transfer_target = m.group(1).strip() if m else None
    cleaned = TRANSFER_RE.sub("", cleaned).strip()
    return cleaned, tool_names, transfer_target


class MedGemmaLlm(BaseLlm):
    """ADK BaseLlm backed by a locally-loaded MedGemma model."""

    model: str = MODEL_ID

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"medgemma.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        import torch

        model_obj, processor = _ModelHolder.get()
        sys_text = _system_text(llm_request)
        messages = _content_to_messages(llm_request.contents or [], sys_text)
        if not messages:
            messages = [{"role": "user", "content": [{"type": "text", "text": "Begin."}]}]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = next(model_obj.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        do_sample = DEFAULT_TEMPERATURE > 0
        # Stop strings cut off the spew the 4B model produces after its real
        # answer at large token budgets - it likes to hallucinate the next
        # user / nurse turn ("For context:[nurse] said:", "**Patient:**", a
        # second `<unused94>thought` block). Stopping at those markers ends
        # the turn at the genuine response.
        stop_strings = [
            "For context:",
            "[nurse] said:",
            "[doctor] said:",
            "[radiologist] said:",
            "**Patient:**",
        ]
        # Forbid the `<unused94>` / `<unused95>` thought-block tokens for
        # the 4B model: it otherwise burns its entire budget inside an
        # open-ended thought (no closing tag) and never produces the user-
        # facing TRIAGE SUMMARY / transfer marker. The 27B model handles
        # thought blocks correctly, so we let those through. Token IDs 100
        # and 101 are `<unused94>` and `<unused95>` in the gemma3 tokenizer.
        bad_words_ids = None
        if "4b" in MODEL_ID.lower():
            bad_words_ids = [[100], [101]]
        with torch.inference_mode():
            generated = model_obj.generate(
                **inputs,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                do_sample=do_sample,
                temperature=DEFAULT_TEMPERATURE if do_sample else 1.0,
                # The 4B model occasionally slides into a degenerate loop
                # (repeating the same checklist paragraph until it hits
                # max_new_tokens) when the conversation history is long.
                # A mild repetition penalty curbs this without distorting
                # output style; no_repeat_ngram_size catches near-repeats
                # the penalty alone misses.
                repetition_penalty=DEFAULT_REPETITION_PENALTY,
                no_repeat_ngram_size=14,
                stop_strings=stop_strings,
                tokenizer=processor.tokenizer,
                bad_words_ids=bad_words_ids,
            )
        prompt_len = inputs["input_ids"].shape[1]
        completion = processor.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True
        )[0].strip()

        cleaned, tool_names, transfer_target = _split_markers(completion)
        parts: list[genai_types.Part] = []
        if cleaned:
            parts.append(genai_types.Part(text=cleaned))
        for tool_name in tool_names:
            parts.append(
                genai_types.Part(
                    function_call=genai_types.FunctionCall(
                        name=tool_name,
                        args={},
                    )
                )
            )
        # If the model asked to invoke a tool, defer the transfer until the
        # next turn (after it has seen the tool result). Otherwise emit the
        # transfer immediately.
        if transfer_target and not tool_names:
            parts.append(
                genai_types.Part(
                    function_call=genai_types.FunctionCall(
                        name="transfer_to_agent",
                        args={"agent_name": transfer_target},
                    )
                )
            )
        if not parts:
            parts.append(genai_types.Part(text=" "))

        yield LlmResponse(
            content=genai_types.Content(role="model", parts=parts),
            partial=False,
            turn_complete=True,
        )
