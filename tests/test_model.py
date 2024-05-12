import io
import os
import unittest

import requests
import torch
from PIL import Image
from transformers import FuyuProcessor

from tests.fixtures.mock_dataset import MockDataset

MODEL_ID = "adept/fuyu-8b"


pretrained_kwargs = {
    "device_map": "auto",
    "torch_dtype": torch.float16,
}

bbox_prompt = "When presented with a box, perform OCR to extract text contained within it. If provided with text, generate the corresponding bounding box.\\n<box>388, 428, 404, 488</box>"
bbox_image_url = (
    "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bbox_sample_image.jpeg"
)
bbox_image_pil = Image.open(io.BytesIO(requests.get(bbox_image_url).content))

# allow for image/word size from env, maybe parameterize later
img_size = tuple(map(int, os.environ.get("IMG_SIZE", "1290,1080").split(",")))
num_words = tuple(map(int, os.environ.get("NUM_WORDS", "100").split(",")))


def print_sample_info(sample, processor, model_key=False):
    if model_key:
        print(model_key)

    inputs = processor(**sample)
    for k, v in inputs.items():
        print(f"{k}.shape: {v.shape if isinstance(v, torch.Tensor) else len(v)}")


def get_model_config_kwargs():
    if os.getenv("FEWGPU"):
        return {
            "num_hidden_layers": 1,
            "text_config": {"model_type": "persimmon", "num_hidden_layers": 1},
        }
    return {}


class TestHF(unittest.TestCase):
    def test_text_extract(self):
        from transformers import FuyuForCausalLM

        processor = FuyuProcessor.from_pretrained(MODEL_ID)
        model = FuyuForCausalLM.from_pretrained(MODEL_ID, **pretrained_kwargs)

        model_inputs = processor(text=bbox_prompt, images=bbox_image_pil).to("cuda")

        generated_tokens = model.generate(**model_inputs, max_new_tokens=10)

        model_outputs = processor.batch_decode(generated_tokens[:, -10:], skip_special_tokens=True)[0]
        prediction = model_outputs.split("\x04", 1)[1] if "\x04" in model_outputs else ""
        self.assertTrue("\x04" in model_outputs)
        self.assertTrue("Williams" in prediction)

    def test_basemodel_fail(self):
        from transformers import FuyuForCausalLM

        processor = FuyuProcessor.from_pretrained(MODEL_ID)
        model = FuyuForCausalLM.from_pretrained(MODEL_ID, **pretrained_kwargs)

        model_inputs = processor(text=bbox_prompt, images=bbox_image_pil).to("cuda")

        with torch.no_grad():
            model_outputs = model(**model_inputs)

        generated_tokens = model.generate(**model_inputs, max_new_tokens=10)

        model_outputs = processor.batch_decode(generated_tokens[:, -10:], skip_special_tokens=True)[0]
        prediction = model_outputs.split("\x04", 1)[1] if "\x04" in model_outputs else ""
        self.assertTrue("\x04" in model_outputs)
        self.assertTrue("Williams" in prediction)


class TestPatchedModel(unittest.TestCase):
    def test_text_extract(self):
        from hf_fuyu.model.modeling_fuyu import FuyuForCausalLM

        processor = FuyuProcessor.from_pretrained(MODEL_ID)
        model = FuyuForCausalLM.from_pretrained(MODEL_ID, **pretrained_kwargs)

        model_inputs = processor(text=bbox_prompt, images=bbox_image_pil).to("cuda")

        generated_tokens = model.generate(**model_inputs, max_new_tokens=10)

        model_outputs = processor.batch_decode(generated_tokens[:, -10:], skip_special_tokens=True)[0]
        prediction = model_outputs.split("\x04", 1)[1] if "\x04" in model_outputs else ""
        self.assertTrue("\x04" in model_outputs)
        self.assertTrue("Williams" in prediction)


class TestForward(unittest.TestCase):
    def run_forward(self, model, ds, processor, n=5):
        for i in range(n):
            sample = ds[i]
            model_inputs = processor(**sample)
            model_inputs["labels"] = model_inputs["input_ids"].clone()
            model_inputs.to(model.device)
            model_outputs = model(**model_inputs)

            self.assertTrue(model_outputs.loss is not None)

    def test_hf(self):
        from transformers import FuyuConfig, FuyuForCausalLM, FuyuProcessor

        model_config = FuyuConfig.from_pretrained(MODEL_ID, **get_model_config_kwargs())
        model = FuyuForCausalLM.from_pretrained(MODEL_ID, device_map="auto", config=model_config)
        processor = FuyuProcessor.from_pretrained(MODEL_ID)
        ds = MockDataset(img_size=img_size, num_words=num_words)
        print_sample_info(ds[0], processor, model_key="hf")
        self.run_forward(model, ds, processor)

    def test_patched(self):
        from hf_fuyu.model.modeling_fuyu import FuyuForCausalLM
        from transformers import FuyuConfig, FuyuProcessor

        ds = MockDataset(img_size=img_size, num_words=num_words)
        model_config = FuyuConfig.from_pretrained(MODEL_ID, **get_model_config_kwargs())
        model = FuyuForCausalLM.from_pretrained(MODEL_ID, device_map="auto", config=model_config)
        processor = FuyuProcessor.from_pretrained(MODEL_ID)
        print_sample_info(ds[0], processor, model_key="patched")
        self.run_forward(model, ds, processor)
