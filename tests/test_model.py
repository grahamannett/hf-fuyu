import io
import unittest

import requests
import torch

from PIL import Image
from transformers import FuyuProcessor

MODEL_ID = "adept/fuyu-8b"


pretrained_kwargs = {
    "device_map": "auto",
    "torch_dtype": torch.float16,
}

bbox_prompt = "When presented with a box, perform OCR to extract text contained within it. If provided with text, generate the corresponding bounding box.\\n<box>388, 428, 404, 488</box>"
bbox_image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bbox_sample_image.jpeg"
bbox_image_pil = Image.open(io.BytesIO(requests.get(bbox_image_url).content))


class TestModel(unittest.TestCase):
    def test_model(self):
        self.assertTrue(True)

    def test_basemodel_fail(self):
        from transformers import FuyuForCausalLM

        processor = FuyuProcessor.from_pretrained(MODEL_ID)
        model = FuyuForCausalLM.from_pretrained(MODEL_ID, **pretrained_kwargs)

        model_inputs = processor(text=bbox_prompt, images=bbox_image_pil).to("cuda")

        generated_tokens = model.generate(**model_inputs, max_new_tokens=10)

        model_outputs = processor.batch_decode(
            generated_tokens[:, -10:], skip_special_tokens=True
        )[0]
        prediction = (
            model_outputs.split("\x04", 1)[1] if "\x04" in model_outputs else ""
        )
        self.assertTrue("\x04" in model_outputs)
        self.assertTrue("Williams" in prediction)

    def test_text_extract(self):
        from hf_fuyu.model.modeling_fuyu import FuyuForCausalLM

        processor = FuyuProcessor.from_pretrained(MODEL_ID)
        model = FuyuForCausalLM.from_pretrained(MODEL_ID, **pretrained_kwargs)

        model_inputs = processor(text=bbox_prompt, images=bbox_image_pil).to("cuda")

        generated_tokens = model.generate(**model_inputs, max_new_tokens=10)

        model_outputs = processor.batch_decode(
            generated_tokens[:, -10:], skip_special_tokens=True
        )[0]
        prediction = (
            model_outputs.split("\x04", 1)[1] if "\x04" in model_outputs else ""
        )
        self.assertTrue("\x04" in model_outputs)
        self.assertTrue("Williams" in prediction)
