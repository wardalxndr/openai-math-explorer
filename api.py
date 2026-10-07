"""
OpenAI Math Explorer backend: hybrid retrieval plus grounded answers.
Kept for local use. The public demo runs on Hugging Face Spaces (app.py).

Endpoints:
  GET /search?q=...&k=3   hybrid keyword+vector search, no LLM needed
  GET /ask?q=...          search plus LLM answer with citations (needs Ollama or Groq)

Run: uvicorn api:app --host 127.0.0.1 --port 8000
Env: OLLAMA_MODEL (default llama3.1), GROQ_API_KEY (optional, enables ?provider=groq)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from retrieval import build_prompt, groq_answer, hybrid, ollama_answer

app = FastAPI(title="OpenAI Math Explorer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/search")
def search(q: str, k: int = 3):
    return {"query": q, "results": hybrid(q, max(1, min(k, 10)))}


@app.get("/ask")
def ask(q: str, provider: str = "ollama"):
    prompt, ctx_chunks = build_prompt(q, 3)
    try:
        answer = groq_answer(prompt) if provider == "groq" else ollama_answer(prompt)
    except Exception as e:  # model not pulled, offline, no key, etc.
        return {
            "query": q,
            "answer": "",
            "error": f"LLM unavailable ({type(e).__name__}). Retrieval still works via /search.",
            "citations": ctx_chunks,
        }
    return {
        "query": q,
        "answer": answer,
        "citations": [
            {"id": c["id"], "title": c["title"], "family": c["family"], "pdf_path": c["pdf_path"]}
            for c in ctx_chunks
        ],
    }
