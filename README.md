# OpenAI Math Explorer - RAG Tanya Jawab 722 Naskah

RAG kecil di atas rilis openai math (722 naskah, 372 family, lisensi Apache 2.0).
Tujuan: jawab pertanyaan soal isi naskah dengan sitasi file dan halaman, plus badge status verifikasi Lean.

Scope awal: `overview.pdf` (41 halaman) + 10 PDF `reasoning_traces` + metadata family.

## Cara jalanin

1. `pip install -r requirements.txt`
2. Clone sumber (sekali aja, berat): `git clone https://github.com/openai/math.git ../openai-math`
3. Generate index: `python scripts/build_math_index.py --src ../openai-math --out public/math-index.json`
4. Buka Explore UI (nyusul): `streamlit run app.py`

## Struktur

- `scripts/build_math_index.py`: PDF jadi chunk JSON
- `public/math-index.json`: index siap search (digenerate, 326 chunk sekitar 359 KB)
- `app.py`: UI Explore + Ask (nyusul)

## Evaluasi (nyusul ala zoomcamp)

- Retrieval: 30 ground truth manual, ukur hit rate dan MRR, bandingkan keyword vs vector
- Jawaban: bandingkan 2 prompt, nilai pakai LLM as a Judge

## Catatan jujur

Banyak naskah belum peer review dan belum formal Lean. Yang belum formal bisa salah.
Aplikasi ini selalu tampilkan sumber dan warning itu.
