import json
import sys

sys.path.insert(0, ".")
from retrieval import (
    COLLECTION,
    HIGHLIGHTS,
    PROMPT,
    hybrid,
    ollama_answer,
    tokens,
)

QUESTIONS = [
    ("Why should I care about the Riemann hypothesis?", "003"),
    ("Which result makes AI computers faster?", "107"),
    ("Which math keeps my Bitcoin safe?", "002"),
]


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
    return context, ranked_hl[0]


out = []
for q, fam in QUESTIONS:
    context, _ = full_context(q)
    top_hl = next(h for h in HIGHLIGHTS if h["family"] == fam)
    a = ollama_answer(PROMPT.format(context=context, question=q))
    out.append(
        {
            "q": q,
            "a": a,
            "citations": [
                {
                    "id": f"highlight-{top_hl['family']}",
                    "title": top_hl["title"],
                    "family": top_hl["family"],
                    "pdf_path": top_hl["pdf_hint"],
                }
            ],
        }
    )
    print("done:", q[:40], "-> family", top_hl["family"])

with open(
    "C:/Users/lexdw/AppData/Local/Temp/opencode/presets.json", "w", encoding="utf-8"
) as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("saved", len(out))
