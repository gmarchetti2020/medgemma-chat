"""Custom ADK BaseLlm wrapper around google/medgemma-1.5-4b-it.

Loads the model once per process (singleton) and shares it across all
agents in the urgent-care app. Translates ADK's LlmRequest into the
gemma3 chat-template format (text + images), runs generation, and
parses an explicit `[[TRANSFER:agent_name]]` marker out of the model
output so we can hand control off via ADK's normal transfer flow even
though MedGemma cannot natively emit structured function calls.
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

MODEL_ID = os.environ.get("MEDGEMMA_MODEL_ID", "google/medgemma-1.5-4b-it")
TRANSFER_RE = re.compile(r"\[\[\s*TRANSFER\s*:\s*([A-Za-z0-9_\-]+)\s*\]\]")
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MEDGEMMA_MAX_NEW_TOKENS", "1024"))
DEFAULT_TEMPERATURE = float(os.environ.get("MEDGEMMA_TEMPERATURE", "0.7"))


class _ModelHolder:
    """Process-wide lazy singleton so the 4B model is loaded once."""

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
                        device_map = "auto"
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
    """Convert ADK Content list to gemma3 chat-template messages."""
    messages: list[dict] = []
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
                out_content.append(
                    {
                        "type": "text",
                        "text": (
                            f"[Tool `{fr.name}` returned: {fr.response}]"
                        ),
                    }
                )
        if out_content:
            messages.append({"role": role, "content": out_content})

    # Gemma3's chat template does not consistently accept a `system` role,
    # so prepend the system text to the first user turn for compatibility.
    if system_text:
        injected = {"type": "text", "text": system_text}
        for i, m in enumerate(messages):
            if m["role"] == "user":
                m["content"] = [injected] + m["content"]
                break
        else:
            messages.insert(0, {"role": "user", "content": [injected]})
    return messages


def _split_transfer(text: str) -> tuple[str, Optional[str]]:
    m = TRANSFER_RE.search(text)
    if not m:
        return text.strip(), None
    target = m.group(1).strip()
    cleaned = TRANSFER_RE.sub("", text).rstrip()
    return cleaned, target


class MedGemmaLlm(BaseLlm):
    """ADK BaseLlm backed by a locally-loaded MedGemma 1.5 4B model."""

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
        with torch.inference_mode():
            generated = model_obj.generate(
                **inputs,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                do_sample=do_sample,
                temperature=DEFAULT_TEMPERATURE if do_sample else 1.0,
            )
        prompt_len = inputs["input_ids"].shape[1]
        completion = processor.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True
        )[0].strip()

        cleaned, transfer_target = _split_transfer(completion)
        parts: list[genai_types.Part] = []
        if cleaned:
            parts.append(genai_types.Part(text=cleaned))
        if transfer_target:
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
