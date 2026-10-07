"""
Vector + hybrid retrieval eval over math-index.json.
Compares keyword vs vector vs hybrid (RRF) on eval/ground_truth.json.

Run: python eval/vector_search.py [--k 5]
Needs: pip install fastembed (downloads a ~100 MB model once, then works offline).
Embeddings are cached in eval/embeddings.npy so reruns are fast.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

try:
    from evaluate import rank as keyword_rank
except ImportError:
    from eval.evaluate import rank as keyword_rank  # type: ignore


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)) @ (
        b / (np.linalg.norm(b) + 1e-9)
    )


def get_embeddings(chunks: list[dict]) -> np.ndarray:
    cache = HERE / "embeddings.npy"
    if cache.exists():
        return np.load(str(cache))
    from fastembed import TextEmbedding

    model = TextEmbedding()  # BAAI/bge-small-en-v1.5, downloads once
    texts = [c["title"] + " " + c["chunk_text"] for c in chunks]
    vecs = np.array(list(model.embed(texts)))
    np.save(str(cache), vecs)
    return vecs


def vector_rank(query: str, chunks: list[dict], vecs: np.ndarray) -> list[str]:
    from fastembed import TextEmbedding

    model = TextEmbedding()
    q = np.array(list(model.embed([query])))[0]
    order = np.argsort(-cosine(vecs, q))
    return [chunks[i]["id"] for i in order]


def rrf_fuse(rank_lists: list[list[str]], c: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in rank_lists:
        for i, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (c + i + 1)
    return sorted(scores, key=lambda cid: -scores[cid])


def evaluate(rank_fn, gt: list[dict], ks: list[int]) -> tuple[dict[int, float], float]:
    hits = {k: 0 for k in ks}
    mrr = 0.0
    for item in gt:
        want = set(item["relevant_ids"])
        ranked = rank_fn(item["question"])
        best = next((i + 1 for i, cid in enumerate(ranked) if cid in want), None)
        if best is None:
            continue
        mrr += 1.0 / best
        for k in ks:
            if best <= k:
                hits[k] += 1
    n = len(gt)
    return ({k: hits[k] / n for k in ks}, mrr / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    ks = sorted({1, 3, args.k})

    chunks = json.loads((HERE / "math-index.json").read_text(encoding="utf-8"))
    gt = json.loads((HERE / "ground_truth.json").read_text(encoding="utf-8"))
    vecs = get_embeddings(chunks)

    kw = lambda q: keyword_rank(q, chunks)  # noqa: E731
    vc = lambda q: vector_rank(q, chunks, vecs)  # noqa: E731
    hy = lambda q: rrf_fuse([vc(q), kw(q)])  # noqa: E731

    for name, fn in [("keyword ", kw), ("vector  ", vc), ("hybrid  ", hy)]:
        hits, mrr = evaluate(fn, gt, ks)
        line = "  ".join(f"hit@{k}={v:.2f}" for k, v in hits.items())
        print(f"{name} {line}  MRR={mrr:.3f}")


if __name__ == "__main__":
    main()
