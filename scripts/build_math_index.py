"""
Build public/math-index.json from openai/math.
Scope kecil dulu: overview.pdf + reasoning_traces (10 PDFs).
Jalankan: python scripts/build_math_index.py --src ../openai-math --out public/math-index.json
"""
import argparse
import json
import re
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

TRACE_META = {
    "ordinary-two-point-correlations": {"family": "007", "subject": "Number theory"},
    "irrationality-exponent-of-pi": {"family": "017", "subject": "Number theory"},
    "symmetric-and-general-mahler-conjectures": {"family": "087", "subject": "Convex geometry"},
    "basic-semidefinite-threshold-np-hardness": {"family": "102", "subject": "Theoretical computer science"},
    "quasipolynomial-arithmetic-progressions": {"family": "159", "subject": "Combinatorics"},
    "kaplansky-direct-finiteness-characteristic-two": {"family": "197", "subject": "Algebra"},
    "mezard-parisi-formula": {"family": "221", "subject": "Probability"},
    "spontaneous-magnetization-quantum-heisenberg-ferromagnet": {"family": "271", "subject": "Mathematical physics"},
    "free-group-factor-isomorphism": {"family": "287", "subject": "Operator algebras"},
    "relativistic-vlasov-maxwell": {"family": "362", "subject": "PDE"},
}

def read_pdf(path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError("pip install pypdf dulu: pip install pypdf")
    reader = PdfReader(str(path))
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)

def clean(text: str) -> str:
    """Normalize PDF extraction artifacts: math alphabets to ASCII,
    exotic spaces to normal spaces, common symbols to plain text."""
    import unicodedata

    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    for a, b in [
        ("−", "-"),
        ("–", "-"),
        ("—", "-"),
        ("≥", ">="),
        ("≤", "<="),
        ("·", "*"),
        ("×", "x"),
        ("’", "'"),
        ("‘", "'"),
        ("“", '"'),
        ("”", '"'),
        ("…", "..."),
    ]:
        text = text.replace(a, b)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk(text: str, size: int = 800) -> list[str]:
    text = clean(text)
    return [text[i:i+size] for i in range(0, len(text), size) if text[i:i+size].strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder hasil git clone openai/math")
    ap.add_argument("--out", default="public/math-index.json")
    args = ap.parse_args()

    src = Path(args.src)
    items = []

    overview = src / "overview.pdf"
    if overview.exists():
        for i, c in enumerate(chunk(read_pdf(overview))[:60]):
            items.append({
                "id": f"overview-{i}",
                "family": "overview",
                "title": "Overview 372 families",
                "subject": "Catalog",
                "pdf_path": "overview.pdf",
                "lean_status": "mixed",
                "chunk_text": c,
            })

    traces = src / "reasoning_traces"
    if traces.exists():
        for pdf in sorted(traces.glob("*.pdf")):
            key = pdf.stem
            meta = TRACE_META.get(key, {"family": "unknown", "subject": "Math"})
            for i, c in enumerate(chunk(read_pdf(pdf))[:30]):
                items.append({
                    "id": f"{key}-{i}",
                    "family": meta["family"],
                    "title": key.replace("-", " "),
                    "subject": meta["subject"],
                    "pdf_path": f"reasoning_traces/{pdf.name}",
                    "lean_status": "unverified, check Lean folder",
                    "chunk_text": c,
                })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: {len(items)} chunks -> {out} ({out.stat().st_size/1024:.1f} KB)")

if __name__ == "__main__":
    main()
