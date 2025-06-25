#!/usr/bin/env -S uv run --script
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from gr_demo_3dgen.groq import groq_create_client, groq_transcribe_audio
from gr_demo_3dgen.utils import np_wav_to_compressed_buffer

load_dotenv()
gr.set_static_paths(paths=[Path.cwd().absolute() / "public"])

groq_client = groq_create_client()

HEAD = """
<script src="/gradio_api/file=public/vad.js"></script>
"""


with gr.Blocks(head=HEAD) as demo:
    gr.Markdown("# Greetings from Gradio!")

    with gr.Row():
        input_audio = gr.Audio(
            label="Voice Command",
            type="numpy",
            streaming=False,
        )
        input_text = gr.Textbox(
            label="Optional Text Guidance",
            placeholder="Optionally tell agent what you sketched. Voice Command will override this.",
            interactive=True,
            lines=2,
        )

        def test(val):
            sr, wav = val
            audio_buf = np_wav_to_compressed_buffer(sr, wav)
            text = groq_transcribe_audio(groq_client, audio_buf)
            return text

        input_audio.input(test, inputs=input_audio, outputs=input_text)

    inp = gr.Textbox(placeholder="What is your name?")
    out = gr.Textbox()

    inp.change(fn=lambda x: f"Welcome, {x}!", inputs=inp, outputs=out)

if __name__ == "__main__":
    demo.launch()
