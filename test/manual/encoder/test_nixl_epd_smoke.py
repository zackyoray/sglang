"""Run one real image request through the existing EPD integration fixture."""

import base64
import importlib.util
import io
import json
from pathlib import Path

import requests
from PIL import Image
from transformers import AutoTokenizer


def load_epd_test_class():
    path = (
        Path(__file__).resolve().parents[2]
        / "registered/disaggregation/test_epd_disaggregation.py"
    )
    spec = importlib.util.spec_from_file_location("epd_disaggregation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TestEPDDisaggregationMooncake


def image_data_url():
    image = Image.new("RGB", (64, 64), (220, 30, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def main():
    fixture = load_epd_test_class()
    fixture.setUpClass()
    try:
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "What is the main color? Answer with one word.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokenizer = AutoTokenizer.from_pretrained(fixture.model, trust_remote_code=True)
        response = requests.post(
            fixture.lb_url + "/generate",
            json={
                "input_ids": tokenizer.encode(prompt),
                "image_data": [image_data_url()],
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["text"]
        if not content.strip():
            raise AssertionError("EPD returned an empty completion")
        if "red" not in content.lower():
            raise AssertionError(
                f"EPD did not identify the red test image: {content!r}"
            )
        print(
            "NIXL EPD image request OK: "
            + json.dumps({"content": content, "usage": payload.get("usage")})
        )
    finally:
        fixture.tearDownClass()


if __name__ == "__main__":
    main()
