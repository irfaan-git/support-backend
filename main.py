import json
import re
import os
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from llama_cpp import Llama

# --- RAG LIBRARIES ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# PHASE 1: THE RAG PIPELINE (ONE-TIME SETUP)
# ==========================================

def extract_pdf_text(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: Could not find {file_path}")
        return ""
    
    print(f"Reading {file_path}...")
    raw_text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text + " "
    except Exception as e:
        print(f"Error reading PDF: {e}")
    
    return re.sub(r'\s+', ' ', raw_text).strip()

def chunk_text(text, chunk_size=60, overlap=15):
    """
    THE CHOPPER: Splits the massive text into smaller, overlapping windows of words.
    Overlap ensures we don't accidentally cut a rule in half.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    print(f"Chopped document into {len(chunks)} distinct chunks.")
    return chunks

def build_vector_db(chunks):
    """
    THE FILING CABINET: Converts the text chunks into a mathematical matrix.
    """
    if not chunks:
        return None, None
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(chunks)
    print("Vector Database built successfully!")
    return vectorizer, tfidf_matrix

def load_prompt_template(file_path):
    if not os.path.exists(file_path):
        return "<start_of_turn>user\n{user_message}\n<end_of_turn>\n<start_of_turn>model\n" 
    with open(file_path, 'r', encoding="utf-8") as f:
        return f.read()

# --- INITIALIZE THE SYSTEM ---
PDF_FILE_NAME = "data/Habytet_POS_Complete_FAQ.pdf" 
PROMPT_FILE_NAME = "prompts/support_agent.txt"

# 1. Extract
RAW_TEXT = extract_pdf_text(PDF_FILE_NAME)
# 2. Chunk
CHUNKS = chunk_text(RAW_TEXT)
# 3. Vectorize (Store in memory)
VECTORIZER, TFIDF_MATRIX = build_vector_db(CHUNKS)
# 4. Load Instructions
PROMPT_TEMPLATE = load_prompt_template(PROMPT_FILE_NAME)

print("Loading local Gemma model into memory... (CPU Stable Mode)")
llm = Llama(
    model_path="./gemma-4-E2B-it-Q4_K_M.gguf",
    n_ctx=2048,        
    n_gpu_layers=0,    
    verbose=False      
)
print("Model loaded successfully!")

# ==========================================
# PHASE 2: REAL-TIME QUERY (USER FLOW)
# ==========================================

class ChatRequest(BaseModel):
    user_message: str

class FeedbackRequest(BaseModel):
    user_message: str
    bot_answer: str

def retrieve_context(query, top_k=3):
    """
    THE SEARCH ENGINE: Converts the user's question into math, compares it 
    against the matrix, and returns the top matching chunks.
    """
    if VECTORIZER is None or TFIDF_MATRIX is None:
        return "No knowledge base available."
    
    # Convert query to vector
    query_vec = VECTORIZER.transform([query])
    
    # Calculate cosine similarity distance
    similarities = cosine_similarity(query_vec, TFIDF_MATRIX).flatten()
    
    # Get indices of the top_k highest scores
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    print("\n--- RAG RETRIEVAL LOG ---")
    print(f"User Query: '{query}'")
    retrieved_chunks = []
    for i, idx in enumerate(top_indices):
        score = similarities[idx]
        chunk = CHUNKS[idx]
        retrieved_chunks.append(chunk)
        print(f"Match {i+1} (Score: {score:.2f}): {chunk[:75]}...")
    print("-------------------------\n")
    
    # Combine the top chunks into a single string for the prompt
    return "\n...\n".join(retrieved_chunks)


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # 1. RETRIEVE relevant chunks instead of the whole PDF
    relevant_context = retrieve_context(request.user_message, top_k=3)
    
    # 2. INJECT only those chunks into the prompt
    formatted_prompt = PROMPT_TEMPLATE.format(
        knowledge_base=relevant_context,
        user_message=request.user_message
    )
    
    # 3. GENERATE answer
    output = llm(
        formatted_prompt,
        max_tokens=350,
        stop=["<end_of_turn>"],
        echo=False
    )
    
    raw_text = output['choices'][0]['text'].strip()
    
    # Robust JSON Extraction
    clean_text = raw_text.replace("```json", "").replace("```", "").replace("<end_of_turn>", "").strip()
    
    try:
        return json.loads(clean_text)
    except Exception:
        json_match = re.search(r'\{.*?\}', clean_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except Exception:
                pass
        
        return {"action": "answer", "reply": raw_text}

@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    return {
        "action": "raise_ticket",
        "title": "Unresolved Support Query",
        "description": f"User query: {request.user_message}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)