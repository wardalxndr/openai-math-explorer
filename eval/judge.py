"""
LLM-as-a-Judge: compare prompt A (fun, benefit-first) vs prompt B (strict factual).
10 sampled ground truths. Candidates and judge both via Gemini free tier.
Needs GEMINI_API_KEY in the machine environment (setx, never in chat).

Run: python eval/judge.py
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval import COLLECTION, HIGHLIGHTS, hybrid, tokens

HERE = Path(__file__).resolve().parent
KEY = os.environ["GEMINI_API_KEY"]
SAMPLE = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]

BASE_INSTRUCT = "Cite sources like [family 017]. If the context lacks the answer, say you do not know. Plain text only, no markdown, no asterisks. Facts stay exact, never invent anything."
PROMPT_A = (
    "Answer ONLY from the context below. "
    "First sentence must be the real-life payoff in plain words. No definitions, no formulas, "
    "no jargon like Dirichlet, L-function, zero-free, exponent, or corank. "
    "Explain like a chill Gen Z friend: smooth, fun, straight to the point. "
    "Keep the whole answer under 90 words. " + BASE_INSTRUCT + "\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
)
PROMPT_B = (
    "Answer ONLY from the context below. Be precise and technical. "
    "Name the exact objects, conditions, and results. Keep under 120 words. "
    + BASE_INSTRUCT
    + "\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
)
JUDGE_PROMPT = """You judge two answers to a math question. The expected answer must come from this chunk:
EXPECTED: {expected}

Question: {question}

Answer A: {a}
Answer B: {b}

Score each 1-5 on: grounded (no invented facts), correct (matches expected), clear (easy to read).
Reply ONLY as JSON: {{"a": {{"grounded": n, "correct": n, "clear": n}}, "b": {{"grounded": n, "correct": n, "clear": n}}, "winner": "a or b"}}"""


def gemini(prompt: str, temperature: float = 0.2, max_tokens: int = 300) -> str:
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
    ).encode()
    last: Exception | None = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key="
                + KEY,
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read())["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            last = e
            import time

            time.sleep(90 if e.code == 429 else 15 * (attempt + 1))
        except Exception as e:
            last = e
            import time

            time.sleep(15 * (attempt + 1))
    raise last if last else RuntimeError("gemini failed")


def context_for(query: str) -> tuple[str, list[dict]]:
    ctx = hybrid(query, 3)
    qs = tokens(query)
    ranked_hl = sorted(
        HIGHLIGHTS,
        key=lambda h: -sum(
            (h["title"] + " " + h["claim"] + " " + h["why"]).lower().count(w) for w in qs
        ),
    )
    why_block = "\n\n".join(
        f"[highlight family {h['family']}] {h['title']}: {h['claim']} Why it matters: {h['why']}"
        for h in ranked_hl[:2]
    )
    context = COLLECTION + "\n\n" + why_block + "\n\n" + "\n\n".join(
        f"[family {c['family']}] {c['title']}: {c['chunk_text'][:400]}" for c in ctx
    )
    return context, ctx


def judge(expected: str, question: str, a: str, b: str) -> dict:
    import re as _re

    last: Exception | None = None
    for _ in range(3):
        try:
            raw = gemini(
                JUDGE_PROMPT.format(
                    expected=expected[:1200], question=question, a=a[:800], b=b[:800]
                ),
                0,
                600,
            )
            m = _re.search(r"\{.*\}", raw, _re.DOTALL)
            verdict = json.loads(m.group(0) if m else raw)
            if verdict.get("winner") in ("a", "b"):
                return verdict
        except Exception as e:
            last = e
    raise last if last else RuntimeError("judge failed")


def main() -> None:
    gt = json.loads((HERE / "ground_truth.json").read_text(encoding="utf-8"))
    chunks = {c["id"]: c for c in json.loads((HERE / "math-index.json").read_text(encoding="utf-8"))}
    out_file = HERE / "judge_results.json"
    done: dict[int, dict] = {}
    if out_file.exists():
        try:
            for r in json.loads(out_file.read_text(encoding="utf-8")).get("results", []):
                done[r["idx"]] = r
        except Exception:
            pass
    results = []
    wins = {"a": 0, "b": 0}
    for idx in SAMPLE:
        if idx in done:
            r = done[idx]
            results.append(r)
            wins[r["verdict"]["winner"]] += 1
            print(f"skip {idx}: winner={r['verdict']['winner']} (cached)", flush=True)
            continue
        item = gt[idx]
        q = item["question"]
        context, _ = context_for(q)
        expected = " ".join(chunks[cid]["chunk_text"][:400] for cid in item["relevant_ids"] if cid in chunks)
        a = gemini(PROMPT_A.format(context=context, question=q))
        b = gemini(PROMPT_B.format(context=context, question=q))
        verdict = judge(expected, q, a, b)
        wins[verdict["winner"]] += 1
        results.append({"idx": idx, "q": q, "a": a, "b": b, "verdict": verdict})
        print(f"done {idx}: winner={verdict['winner']}", flush=True)
        out_file.write_text(
            json.dumps({"wins": wins, "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print("WINS:", wins)


if __name__ == "__main__":
    main()
