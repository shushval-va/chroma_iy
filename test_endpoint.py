"""Test script for Chroma + PuLID RunPod Serverless endpoint.

Usage:
    export RUNPOD_API_KEY=your_key
    python test_endpoint.py
"""

import base64
import os
import sys
import time

import requests

ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "YOUR_ENDPOINT_ID")
API_KEY = os.environ.get("RUNPOD_API_KEY", "")
FACE_IMAGE_PATH = os.environ.get("FACE_IMAGE", "test_face.jpg")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "output.png")

BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def run_test():
    if not API_KEY:
        print("ERROR: Set RUNPOD_API_KEY environment variable")
        sys.exit(1)
    if not os.path.exists(FACE_IMAGE_PATH):
        print(f"ERROR: Face image not found: {FACE_IMAGE_PATH}")
        sys.exit(1)

    face_b64 = encode_image(FACE_IMAGE_PATH)
    payload = {
        "input": {
            "face_image": face_b64,
            "prompt": "a beautiful woman in a red dress, full body, fashion photography, studio lighting",
            "width": 768,
            "height": 1024,
            "num_steps": 26,
            "guidance_scale": 0.0,
            "id_weight": 1.0,
            "seed": 42,
        }
    }

    print(f"Sending request to endpoint {ENDPOINT_ID}...")
    resp = requests.post(f"{BASE_URL}/run", json=payload, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    job_id = data["id"]
    print(f"Job submitted: {job_id}")

    # Poll for result
    while True:
        time.sleep(3)
        resp = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        print(f"  status: {status}")

        if status == "COMPLETED":
            output = data["output"]
            img_b64 = output["image"]
            seed = output.get("seed", "?")
            img_bytes = base64.b64decode(img_b64)
            with open(OUTPUT_PATH, "wb") as f:
                f.write(img_bytes)
            print(f"Done! Seed={seed}, saved to {OUTPUT_PATH}")
            break
        elif status == "FAILED":
            print(f"FAILED: {data.get('error', data)}")
            break
        elif status in ("IN_QUEUE", "IN_PROGRESS"):
            continue
        else:
            print(f"Unknown status: {status}")
            print(data)
            break


if __name__ == "__main__":
    run_test()
