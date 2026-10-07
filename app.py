"""Gradio demo for Hugging Face Spaces (free ZeroGPU).
Uses hybrid retrieval plus Groq (set GROQ_API_KEY in Space Secrets).
"""
import os

import gradio as gr
import spaces

from retrieval import build_prompt, groq_answer, hybrid

EXAMPLES = [
    "What is the irrationality exponent of pi?",
    "Which papers discuss the Riemann zeta function?",
    "What is the Lean verification status?",
]


@spaces.GPU
def answer(question: str) -> str:
    prompt, ctx = build_prompt(question)
    if not os.environ.get("GROQ_API_KEY"):
        lines = [f"Family {c['family']}: {c['title']}" for c in ctx]
        return (
            "LLM key missing (set GROQ_API_KEY in Space Secrets). "
            "Top retrieved chunks:\n- " + "\n- ".join(lines)
            + "\n\nNote: claims come from the OpenAI overview and are not independently verified."
        )
    try:
        text = groq_answer(prompt)
    except Exception as e:
        return f"LLM error ({type(e).__name__}). Try again later."
    cites = "\n".join(
        f"- [{c['id']}] {c['title']} (family {c['family']}, {c['pdf_path']})"
        for c in ctx
    )
    return f"{text}\n\nSources:\n{cites}"


demo = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(label="Ask about the 722 manuscripts", placeholder="e.g. What is the irrationality exponent of pi?"),
    outputs=gr.Markdown(label="Answer with citations"),
    title="OpenAI Math Explorer",
    description="Grounded Q&A over the openai/math release (overview + 10 reasoning traces, 326 chunks). Claims as stated by OpenAI, not independently verified.",
    examples=EXAMPLES,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
