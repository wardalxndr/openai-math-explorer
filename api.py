"""
OpenAI Math Explorer backend: hybrid retrieval plus grounded answers.

Endpoints:
  GET /search?q=...&k=3   hybrid keyword+vector search, no LLM needed
  GET /ask?q=...          search plus LLM answer with citations (needs Ollama)

Run: uvicorn api:app --host 127.0.0.1 --port 8000
Env: OLLAMA_MODEL (default llama3.1), GROQ_API_KEY (optional, enables ?provider=groq)
"""
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

HERE = Path(__file__).resolve().parent
EVAL = HERE / "eval"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

STOP = set(
    "what which whose whom is are was were be been the a an of for in on to and or with about how does do did does it its this that these those from by at as".split()
)

app = FastAPI(title="OpenAI Math Explorer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

CHUNKS: list[dict] = json.loads((EVAL / "math-index.json").read_text(encoding="utf-8"))
VECS = np.load(str(EVAL / "embeddings.npy"))
IDS = [c["id"] for c in CHUNKS]

_model = None


def embed(texts: list[str]) -> np.ndarray:
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding()
    return np.array(list(_model.embed(texts)))


def tokens(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


def keyword_rank(query: str) -> list[str]:
    qs = tokens(query)
    docs = [(c["title"] + " " + c["chunk_text"]).lower() for c in CHUNKS]
    scored = sorted(
        ((sum(d.count(w) for w in qs), cid) for d, cid in zip(docs, IDS)),
        key=lambda x: -x[0],
    )
    return [cid for _, cid in scored]


def vector_rank(query: str) -> list[str]:
    q = embed([query])[0]
    sims = (VECS / (np.linalg.norm(VECS, axis=1, keepdims=True) + 1e-9)) @ (
        q / (np.linalg.norm(q) + 1e-9)
    )
    return [IDS[i] for i in np.argsort(-sims)]


def hybrid(query: str, k: int = 3) -> list[dict]:
    scores: dict[str, float] = {}
    for ranked in (vector_rank(query), keyword_rank(query)):
        for i, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + i + 1)
    top = sorted(scores, key=lambda cid: -scores[cid])[:k]
    by_id = {c["id"]: c for c in CHUNKS}
    return [by_id[cid] for cid in top]


PROMPT = """Answer ONLY from the context below. Cite sources like [family 017].
If the context lacks the answer, say you do not know.
Keep the answer under 120 words, plain language.

Context:
{context}

Question: {question}
Answer:"""


def ollama_answer(prompt: str) -> str:
    body = json.dumps(
        {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    ).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["response"]


def groq_answer(prompt: str) -> str:
    key = os.environ["GROQ_API_KEY"]
    body = json.dumps(
        {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


@app.get("/search")
def search(q: str, k: int = 3):
    return {"query": q, "results": hybrid(q, max(1, min(k, 10)))}


@app.get("/ask")
def ask(q: str, provider: str = "ollama"):
    ctx_chunks = hybrid(q, 3)
    context = "\n\n".join(
        f"[family {c['family']}] {c['title']}: {c['chunk_text'][:900]}"
        for c in ctx_chunks
    )
    prompt = PROMPT.format(context=context, question=q)
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
