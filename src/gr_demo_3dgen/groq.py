from __future__ import annotations

import os

import groq

# https://console.groq.com/docs/speech-to-text
STT_MODEL = "distil-whisper-large-v3-en"


def groq_create_client():
    """Initialize groq client."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    return groq.Client(api_key=api_key)


def groq_transcribe_audio(client: groq.Client, audio_buffer):
    """Transcribe audio using the groq client."""
    resp = client.audio.transcriptions.create(
        model=STT_MODEL,
        # my compression util uses ogg (opus).
        file=("audio.ogg", audio_buffer),
        language="en",
        response_format="verbose_json",
    )

    return resp.text
