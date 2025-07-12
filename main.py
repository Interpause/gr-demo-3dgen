#!/usr/bin/env -S uv run --script
import logging
from datetime import datetime
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from gradio_log import Log as GrLog

from gr_demo_3dgen.comfy import generate_3d_prompt
from gr_demo_3dgen.groq import (
    groq_create_client,
    groq_describe_image,
    groq_transcribe_audio,
)
from gr_demo_3dgen.utils import np_wav_to_compressed_buffer, setup_logging

load_dotenv()
gr.set_static_paths(paths=[Path.cwd().absolute() / "public"])
groq_client = groq_create_client()

HEAD = """
<script src="/gradio_api/file=public/vad.js"></script>
"""
OUTPUT_DIR = Path.cwd().absolute() / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "demo.log"
Path(LOG_FILE).touch(exist_ok=True)
setup_logging(LOG_FILE)
log = logging.getLogger("app")

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
            button_submit = gr.Button(
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

    with gr.Row(height="50vh"):
        GrLog(
            str(LOG_FILE),
            label="logs",
            dark=True,
            xterm_font_size=12,
            height="100%",
        )
        preview_3d = gr.Model3D(
            label="3D Model Preview",
            interactive=False,
            height="100%",
        )

    @gr.on([input_audio.input], inputs=input_audio, outputs=input_text)
    def on_voice_input(val):
        if val is None:
            return input_text.value
        sr, wav = val
        audio_buf = np_wav_to_compressed_buffer(sr, wav)
        text = groq_transcribe_audio(groq_client, audio_buf)
        return text

    @gr.on(
        [button_submit.click], inputs=[input_text, input_sketch], outputs=preview_prompt
    )
    def on_submit(text, sketch):
        text = text.strip()
        if len(text) == 0:
            text = None
        prompt = groq_describe_image(groq_client, sketch["composite"], description=text)
        return prompt

    @gr.on(
        [preview_prompt.change],
        inputs=[preview_prompt, input_sketch],
        outputs=preview_3d,
    )
    def on_prompt_change(prompt, sketch):
        gen = generate_3d_prompt(sketch["composite"], prompt.strip())
        raw_file = None
        for status, msg in gen:
            if not status:
                # log status messages
                log.info(msg)
            else:
                assert isinstance(msg, bytes), "Expected raw file data."
                raw_file = msg
                break
        if raw_file is None:
            raise gr.Error("Failed to generate 3D model.")

        # Save the file to the output directory
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.glb"
        output_path = OUTPUT_DIR / filename
        with open(output_path, "wb") as f:
            f.write(raw_file)

        return str(output_path)

    log.info("Demo initialized successfully.")


if __name__ == "__main__":
    demo.launch(share=True)
