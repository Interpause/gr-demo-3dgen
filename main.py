#!/usr/bin/env -S uv run --script
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from gr_demo_3dgen.groq import (
    groq_create_client,
    groq_describe_image,
    groq_transcribe_audio,
)
from gr_demo_3dgen.utils import np_wav_to_compressed_buffer

load_dotenv()
gr.set_static_paths(paths=[Path.cwd().absolute() / "public"])

groq_client = groq_create_client()

HEAD = """
<script src="/gradio_api/file=public/vad.js"></script>
"""


with gr.Blocks(head=HEAD) as demo:
    gr.Markdown("# SDS Group 3 3DGen Props Demo")

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Optional Text Guidance",
                placeholder="Optionally tell agent what you sketched. Voice Command will override this.",
                interactive=True,
                lines=3,
            )
            preview_prompt = gr.Textbox(
                label="Prompt Preview",
                placeholder="The prompt that will be sent to the AI art model.",
                interactive=False,
                lines=3,
            )
        with gr.Column():
            input_audio = gr.Audio(
                label="Voice Command",
                sources=["microphone", "upload"],
                type="numpy",
                streaming=False,
            )
            submit_button = gr.Button(
                "Generate 3D Model",
                variant="primary",
                elem_id="submit-button",
            )

    input_sketch = gr.ImageEditor(
        label="Sketch",
        image_mode="RGB",
        type="pil",
        canvas_size=(768, 768),
        layers=False,
        height="50vh",
    )

    @gr.on([input_audio.input], inputs=input_audio, outputs=input_text)
    def on_voice_input(val):
        sr, wav = val
        audio_buf = np_wav_to_compressed_buffer(sr, wav)
        text = groq_transcribe_audio(groq_client, audio_buf)
        return text

    @gr.on(
        [submit_button.click], inputs=[input_text, input_sketch], outputs=preview_prompt
    )
    def on_submit(text, sketch):
        text = text.strip()
        if len(text) == 0:
            text = None
        prompt = groq_describe_image(groq_client, sketch["composite"], description=text)
        return prompt

    @gr.on([preview_prompt.change], inputs=[preview_prompt, input_sketch])
    def on_prompt_change(prompt, sketch):
        pass


if __name__ == "__main__":
    demo.launch()
