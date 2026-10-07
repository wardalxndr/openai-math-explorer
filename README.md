---
title: OpenAI Math Explorer
emoji: 🐨
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 6.29.1
app_file: app.py
pinned: false
---

# OpenAI Math Explorer - RAG over 722 Manuscripts

A small RAG on top of the openai math release (722 manuscripts, 372 families, Apache 2.0 license).
Goal: answer questions about the manuscripts with file and page citations, plus a Lean verification status badge.

Initial scope: `overview.pdf` (41 pages) + 10 PDFs in `reasoning_traces` + family metadata.

## Quickstart

1. `pip install -r requirements.txt`
2. Clone the source (one time, heavy): `git clone https://github.com/openai/math.git ../openai-math`
3. Generate the index: `python scripts/build_math_index.py --src ../openai-math --out public/math-index.json`
4. Open the Explore UI (coming soon): `streamlit run app.py`

## Structure

- `scripts/build_math_index.py`: turns PDFs into chunked JSON
- `public/math-index.json`: generated search-ready index (326 chunks, about 359 KB)
- `app.py`: Explore + Ask UI (coming soon)

## Evaluation (coming soon, zoomcamp style)

- Retrieval: 30 hand-made ground truths, measure hit rate and MRR, compare keyword vs vector
- Answers: compare 2 prompts, score with LLM as a Judge

## Honest note

Many manuscripts are not peer reviewed and not Lean formalized yet. Unformalized ones can be wrong.
This app always shows sources and that warning.
