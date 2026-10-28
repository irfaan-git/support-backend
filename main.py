import json
import re
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pypdf import PdfReader
from llama_cpp import Llama

app = FastAPI(title="Support Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The GGUF weights are gitignored — see README.md for where to get them.
MODEL_PATH = "./gemma-4-E2B-it-Q4_K_M.gguf"

print("Loading local Gemma model into memory...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=8192,
    n_gpu_layers=-1,
    flash_attn=True,
    verbose=False
)
print("Model loaded successfully!")


def load_text_file(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {file_path}")
    return path.read_text(encoding="utf-8")


def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def parse_model_json(raw: str):
    """Pull the JSON object out of whatever the model wrote.

    Gemma reliably wraps its reply in a markdown fence:

        ```json
        {"action": "raise_ticket", ...}
        ```

    A plain `json.loads` on that always raises. That is why every reply used
    to fall through to the "answer" fallback: `raise_ticket` could never
    reach the caller, and the model's generated ticket title was discarded.

    Returns the parsed dict, or None if there is genuinely no JSON in there.
    """
    text = raw.strip()

    # 1. strip a leading ```json / ``` and a trailing ```
    text = re.sub(r"^\s*```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    # 2. fall back to the first {...} block, in case the model added prose
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


PROMPT_TEMPLATE = load_text_file("prompts/support_agent.txt")
KNOWLEDGE_BASE = extract_pdf_text("data/Support Portal Flow.pdf")


class ChatRequest(BaseModel):
    user_message: str


class FeedbackRequest(BaseModel):
    user_message: str
    bot_answer: str


@app.get("/")
async def index():
    """The chat page — open http://127.0.0.1:8000 in a browser."""
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "model": Path(MODEL_PATH).name}


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    formatted_prompt = PROMPT_TEMPLATE.format(
        knowledge_base=KNOWLEDGE_BASE,
        user_message=request.user_message
    )

    output = llm(
        formatted_prompt,
        max_tokens=350,
        stop=["<end_of_turn>"],
        echo=False
    )

    raw_text = output['choices'][0]['text'].strip()
    data = parse_model_json(raw_text)

    if data is None:
        # No JSON at all — treat the whole reply as the answer.
        return {"action": "answer", "reply": raw_text}

    if data.get("action") == "raise_ticket":
        return {
            "action": "raise_ticket",
            "title": data.get("title", "Unresolved Support Query"),
            "description": data.get("description", request.user_message),
        }

    return {"action": "answer", "reply": data.get("reply", raw_text)}


@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    escalate_prompt = f"""<start_of_turn>system
You are a ticket generation assistant. The user was not satisfied with the support answer.
Generate a structured support ticket.
Respond ONLY in JSON:
{{
  "title": "<3 to 6 word title>",
  "description": "<Concise bug description based on query: '{request.user_message}'>"
}}
<end_of_turn>
<start_of_turn>model
"""
    output = llm(
        escalate_prompt,
        max_tokens=250,
        stop=["<end_of_turn>"],
        echo=False
    )
    raw_text = output['choices'][0]['text'].strip()
    ticket_data = parse_model_json(raw_text)

    if ticket_data is None:
        return {
            "action": "raise_ticket",
            "title": "Customer Support Escalation",
            "description": f"Query: {request.user_message}",
        }

    return {
        "action": "raise_ticket",
        "title": ticket_data.get("title", "Unresolved Support Query"),
        "description": ticket_data.get("description", request.user_message),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
