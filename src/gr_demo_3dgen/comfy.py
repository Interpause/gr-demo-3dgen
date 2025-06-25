# Referenced from: https://www.viewcomfy.com/blog/building-a-production-ready-comfyui-api
import io
import json
import time
from pathlib import Path

import imagehash
import requests
from PIL import Image
from pydantic import BaseModel
from websocket import WebSocket

import gr_demo_3dgen.prompts

PROMPT_DIR = Path(gr_demo_3dgen.prompts.__file__).parent.absolute()
PROMPT_NAME = "sketch23d_api_faster.json"
SERVER_ADDRESS = "http://nixrobo.home.arpa:8187"
WS_ADDRESS = "ws://nixrobo.home.arpa:8187"
CLIENT_ID = "literally_placeholder"


class ImageMetadata(BaseModel):
    name: str
    subfolder: str
    type: str


class PromptQueueResponse(BaseModel):
    prompt_id: str
    number: int
    node_errors: dict


class HistoryResponse(BaseModel):
    prompt: list
    outputs: dict
    status: dict
    meta: dict


def create_status_ws():
    ws = WebSocket()
    ws.connect(f"{WS_ADDRESS}/ws?clientId={CLIENT_ID}")
    return ws


def track_progress(ws: WebSocket):
    while True:
        try:
            msg = json.loads(ws.recv())
            if msg["type"] == "progress":
                yield f"Progress: {msg['data']['value']}/{msg['data']['max']}"
            elif msg["type"] == "executing":
                yield f"Executing node: {msg['data']['node']}"
            elif msg["type"] == "execution_cached":
                yield f"Cached execution: {msg['data']}"
            elif msg["type"] == "executed":
                yield f"Executed node: {msg['data']['node']}"
                # TODO: This is very workflow dependent atm.
                if msg["data"]["node"] == "154":
                    yield "Generation complete."
                    yield True

        except Exception as e:
            yield f"Error processing message: {e}"
            yield False


def queue_prompt(image_metadata: ImageMetadata, sketch_description: str):
    with open(PROMPT_DIR / PROMPT_NAME, "r") as file:
        prompt = json.load(file)

    # TODO: This is very workflow dependent atm.
    prompt["196"]["inputs"]["image"] = image_metadata.name
    prompt["192"]["inputs"]["text"] = sketch_description

    data = {"prompt": prompt, "client_id": CLIENT_ID}
    headers = {"Content-Type": "application/json"}
    resp = requests.post(f"{SERVER_ADDRESS}/prompt", json=data, headers=headers)
    obj = PromptQueueResponse.model_validate_json(resp.content)
    return obj


def get_history(prompt_id: str):
    resp = requests.get(f"{SERVER_ADDRESS}/history/{prompt_id}")
    obj = resp.json()
    obj = HistoryResponse.model_validate(obj[prompt_id])
    return obj


def get_file(filename: str, subfolder: str, folder_type: str):
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    resp = requests.get(f"{SERVER_ADDRESS}/view", params=params)
    return resp.content


def upload_image(image: Image.Image):
    buf = io.BytesIO()
    image.save(buf, format="webp", quality=100)
    buf.seek(0)
    filename = f"{imagehash.phash(image, hash_size=16)}.webp"

    files = {"image": (filename, buf, "image/webp")}
    data = {"type": "input", "overwrite": "false"}
    url = f"{SERVER_ADDRESS}/upload/image"
    resp = requests.post(url, files=files, data=data)

    obj = ImageMetadata.model_validate_json(resp.content)
    return obj


def generate_3d_prompt(image: Image.Image, sketch_description: str):
    ws = create_status_ws()

    img_meta = upload_image(image)
    prompt_meta = queue_prompt(img_meta, sketch_description)

    try:
        for status in track_progress(ws):
            if isinstance(status, str):
                yield False, status
            elif isinstance(status, bool):
                if status:
                    break
                else:
                    yield False, "An error occurred during generation."
                    return
    finally:
        ws.close()

    time.sleep(1)
    hist_data = get_history(prompt_meta.prompt_id)
    # TODO: This is very workflow dependent atm.
    filename = hist_data.outputs["154"]["result"][0]
    raw_file = get_file(filename, "3D", "output")
    yield True, raw_file
