# medgemma-chat — urgent-care multi-agent demo

A local Google ADK application that simulates a small urgent-care visit using
three agents, all backed by a single in-process copy of
[`google/medgemma-1.5-4b-it`](https://huggingface.co/google/medgemma-1.5-4b-it):

1. **Nurse** — performs triage, asks for chief complaint and symptoms, and
   collects vital signs (the patient types the values manually since there is
   no physical sensor). Hands off to the doctor.
2. **Doctor** — reads the nurse's notes, runs a focused diagnostic dialogue,
   may refer to the radiologist for imaging, and finally writes an
   assessment-and-plan with a prescription.
3. **Radiologist** — asks the patient to upload a chest/abdominal X-ray or
   CT image, reads it, and returns a structured report to the doctor.

Hand-offs use the marker `[[TRANSFER:<agent_name>]]`, which the custom
`MedGemmaLlm` wrapper rewrites into an ADK `transfer_to_agent` function call.

> **Educational simulation only.** Not a medical device, not for clinical use.

## Layout

```
medgemma-chat/
├── agents/
│   └── urgent_care/
│       ├── __init__.py        # ADK entry-point package
│       ├── agent.py           # defines nurse / doctor / radiologist + root_agent
│       ├── medgemma_llm.py    # custom BaseLlm wrapper around MedGemma 1.5 4B
│       └── prompts.py         # role instructions
├── requirements.txt
└── README.md
```

## Prerequisites

- The `adk` conda env (already provisioned). It ships with `google-adk`,
  `transformers`, `torch`, and `pillow`.
- A CUDA-capable GPU is **strongly recommended**. The model is 4.3 B params;
  on CPU, generation is minutes-per-reply.
- A Hugging Face account with access to
  [`google/medgemma-1.5-4b-it`](https://huggingface.co/google/medgemma-1.5-4b-it).
  The model is gated — accept the license on the model page first.
- A Hugging Face token. Either log in once (`huggingface-cli login`) or set
  `HF_TOKEN=<your token>` in the env that runs `adk web`.

## Run

```bash
conda activate adk

# (one-time) authenticate so transformers can download the gated model
huggingface-cli login   # or:  export HF_TOKEN=hf_xxx

# launch the ADK web UI from the repo root
cd /home/gimarchetti/dev/medgemma-chat
adk web agents
```

Open the printed URL (defaults to <http://127.0.0.1:8000>), pick
`urgent_care` from the agent dropdown, and start a session by saying hello —
the nurse will take over from there.

The first request triggers the model download and weight load (a few GB);
subsequent requests reuse the in-memory model.

## Tunables (env vars)

| Variable | Default | What it does |
|---|---|---|
| `MEDGEMMA_MODEL_ID` | `google/medgemma-1.5-4b-it` | HF model id to load |
| `MEDGEMMA_MAX_NEW_TOKENS` | `1024` | per-turn generation cap |
| `MEDGEMMA_TEMPERATURE` | `0.7` | sampling temperature; `0` disables sampling |
| `HF_TOKEN` | _unset_ | token for gated model download |

## How the hand-offs work

ADK's normal multi-agent flow uses a `transfer_to_agent` tool that the model
emits as a structured function call. MedGemma can't reliably do that, so each
role's prompt instructs it to end with a literal marker like
`[[TRANSFER:doctor]]`. The wrapper in `medgemma_llm.py` strips the marker
from the text and synthesizes a proper `FunctionCall` part, so ADK then runs
its standard transfer flow (parent ↔ child or to a peer sub-agent).

Topology:

- `nurse` is the root agent.
- `doctor` is `nurse.sub_agents[0]`, with `disallow_transfer_to_parent=True`
  so triage never reverses.
- `radiologist` is `doctor.sub_agents[0]`. After the report, transferring
  back to the doctor uses ADK's standard parent-transfer.

## Limitations / things to note

- One model instance, one inference at a time. This is a single-user demo.
- No streaming yet — each turn returns one final response.
- The "labs" referenced by the doctor are listed in the plan but never
  actually executed; this app only wires up imaging via the radiologist.
- Image upload uses ADK Web's built-in attachment control; PDFs and other
  non-image MIME types are ignored by the wrapper.
