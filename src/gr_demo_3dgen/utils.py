import io

import av
import numpy as np


def np_wav_to_compressed_buffer(sample_rate: int, wav: np.ndarray):
    """Compress raw audio and store inside a file buffer."""
    # Some assumptions about audio passed from gradio:
    assert wav.ndim == 2 and wav.shape[1] == 2
    assert wav.dtype == np.int16

    # Groq downsamples to 16kHz mono, so we compress to that to save bandwidth.
    # Balance between file size (upload speed) and decode latency.
    out_rate = 16000
    frame_size = 120  # 120ms max supported by Opus.
    bitrate = 16000  # 16kbps, good enough quality for speech.

    buf = io.BytesIO()
    frame_size = sample_rate // 1000 * frame_size
    container = av.open(buf, mode="w", format="ogg")
    resampler = av.AudioResampler(
        format="s16", layout="mono", rate=out_rate, frame_size=frame_size
    )

    stream = container.add_stream(
        "libopus", rate=out_rate, bit_rate=bitrate, layout="mono"
    )

    for i in range(0, len(wav), frame_size):
        chunk = np.ascontiguousarray(wav[i : i + frame_size].T)
        frame = av.AudioFrame.from_ndarray(chunk, format="s16p", layout="stereo")
        frame.rate = sample_rate
        frames = resampler.resample(frame)

        for frm in frames:
            container.mux(stream.encode(frm))

    # Flush all packets.
    container.mux(stream.encode())

    container.close()
    buf.seek(0)

    # with open("test.ogg", "wb") as f:
    #     f.write(buf.getbuffer())
    # buf.seek(0)  # Reset buffer position for reading.

    return buf
