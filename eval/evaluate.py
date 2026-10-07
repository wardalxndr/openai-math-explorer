"""
Baseline retrieval eval: keyword search over math-index.json.
Measures hit rate and MRR against eval/ground_truth.json.

Run: python eval/evaluate.py [--k 5]
No dependencies, stdlib only.
"""
import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
STOP = set("what which whose whom is are was were be been the a an of for in on to and or with about how does do did does it its this that these those from by at as".split())


def tokens(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


def score(query: str, doc: str) -> int:
    qs = tokens(query)
    if not qs:
        return 0
    text = doc.lower()
    return sum(text.count(w) for w in qs)


def rank(query: str, chunks: list[dict]) -> list[str]:
    scored = sorted(
        ((score(query, c["title"] + " " + c["chunk_text"]), c["id"]) for c in chunks),
        key=lambda x: -x[0],
    )
    return [cid for _, cid in scored]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    chunks = json.loads((HERE / "math-index.json").read_text(encoding="utf-8"))
    gt = json.loads((HERE / "ground_truth.json").read_text(encoding="utf-8"))

    hits = {1: 0, 3: 0, args.k: 0}
    mrr = 0.0
    misses: list[str] = []
    for item in gt:
        want = set(item["relevant_ids"])
        ranked = rank(item["question"], chunks)
        positions = [i + 1 for i, cid in enumerate(ranked) if cid in want]
        best = min(positions) if positions else None
        if best is None:
            misses.append(item["question"])
            continue
        mrr += 1.0 / best
        for k in hits:
            if best <= k:
                hits[k] += 1

    n = len(gt)
    print(f"questions: {n} | chunks: {len(chunks)}")
    for k in sorted(hits):
        print(f"hit@{k}: {hits[k]}/{n} = {hits[k]/n:.2f}")
    print(f"MRR: {mrr/n:.3f}")
    if misses:
        print(f"\nmissed ({len(misses)}):")
        for q in misses:
            print(f"  - {q}")


if __name__ == "__main__":
    main()
