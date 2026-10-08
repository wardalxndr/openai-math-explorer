"""
Shared retrieval for OpenAI Math Explorer.
Used by api.py (local FastAPI) and app.py (Gradio Space).
"""
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE / "eval"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

STOP = set(
    "what which whose whom is are was were be been the a an of for in on to and or with about how does do did does it its this that these those from by at as".split()
)

CHUNKS: list[dict] = json.loads((EVAL / "math-index.json").read_text(encoding="utf-8"))
IDS = [c["id"] for c in CHUNKS]
BY_ID = {c["id"]: c for c in CHUNKS}

_model = None
_vecs: np.ndarray | None = None


def get_vecs() -> np.ndarray:
    """Load cached embeddings or compute them on first run (e.g. fresh Space)."""
    global _vecs
    if _vecs is not None:
        return _vecs
    cache = EVAL / "embeddings.npy"
    if cache.exists():
        _vecs = np.load(str(cache))
        return _vecs
    texts = [c["title"] + " " + c["chunk_text"] for c in CHUNKS]
    _vecs = embed(texts)
    try:
        np.save(str(cache), _vecs)
    except OSError:
        pass
    return _vecs

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
    vecs = get_vecs()
    q = embed([query])[0]
    sims = (vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)) @ (
        q / (np.linalg.norm(q) + 1e-9)
    )
    return [IDS[i] for i in np.argsort(-sims)]


def hybrid(query: str, k: int = 3) -> list[dict]:
    scores: dict[str, float] = {}
    for ranked in (vector_rank(query), keyword_rank(query)):
        for i, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + i + 1)
    top = sorted(scores, key=lambda cid: -scores[cid])[:k]
    return [BY_ID[cid] for cid in top]


PROMPT = """Answer ONLY from the context below. Cite sources like [family 017].
If the context lacks the answer, say you do not know.
Keep the answer under 120 words, plain language.

Context:
{context}

Question: {question}
Answer:"""


def build_prompt(query: str, k: int = 3) -> tuple[str, list[dict]]:
    ctx = hybrid(query, k)
    context = "\n\n".join(
        f"[family {c['family']}] {c['title']}: {c['chunk_text'][:900]}" for c in ctx
    )
    return PROMPT.format(context=context, question=query), ctx


def ollama_answer(prompt: str) -> str:
    body = json.dumps(
        {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    ).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["response"]


def groq_answer(prompt: str) -> str:
    key = os.environ["GROQ_API_KEY"]
    body = json.dumps(
        {
            "model": "qwen/qwen3.8-27b",
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
