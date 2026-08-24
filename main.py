import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from llama_cpp import Llama

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading local Gemma model into memory...")
llm = Llama(
    model_path="./gemma-4-E2B-it-Q4_K_M.gguf",
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

PROMPT_TEMPLATE = load_text_file("prompts/support_agent.txt")
KNOWLEDGE_BASE = extract_pdf_text("data/Support Portal Flow.pdf")

class ChatRequest(BaseModel):
    user_message: str

class FeedbackRequest(BaseModel):
    user_message: str
    bot_answer: str

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
    try:
        return json.loads(raw_text)
    except Exception:
        return {"action": "answer", "reply": raw_text}

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
    try:
        ticket_data = json.loads(raw_text)
        return {
            "action": "raise_ticket",
            "title": ticket_data.get("title", "Unresolved Support Query"),
            "description": ticket_data.get("description", request.user_message)
        }
    except Exception:
        return {
            "action": "raise_ticket",
            "title": "Customer Support Escalation",
            "description": f"Query: {request.user_message}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)