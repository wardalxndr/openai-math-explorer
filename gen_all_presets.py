import json
import sys

sys.path.insert(0, ".")
from retrieval import COLLECTION, HIGHLIGHTS, PROMPT, hybrid, ollama_answer, tokens

START = int(sys.argv[1]) if len(sys.argv) > 1 else 0
END = int(sys.argv[2]) if len(sys.argv) > 2 else 30

GT = json.loads(open("eval/ground_truth.json", encoding="utf-8").read())
BY_ID = {}
from retrieval import BY_ID as _B

OUT = "C:/Users/lexdw/AppData/Local/Temp/opencode/presets30.json"


def full_context(query: str, k: int = 3):
    ctx = hybrid(query, k)
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
    return context


existing = []
try:
    existing = json.loads(open(OUT, encoding="utf-8").read())
except FileNotFoundError:
    pass

for i, item in enumerate(GT[START:END], start=START):
    q = item["question"]
    ctx = hybrid(q, 3)
    a = ollama_answer(PROMPT.format(context=full_context(q), question=q))
    cites = [
        {
            "id": _B[cid]["id"],
            "title": _B[cid]["title"],
            "family": _B[cid]["family"],
            "pdf_path": _B[cid]["pdf_path"],
        }
        for cid in item["relevant_ids"]
        if cid in _B
    ]
    existing.append({"q": q, "a": a, "citations": cites})
    print("done", i, q[:45])
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
print("saved", len(existing))
