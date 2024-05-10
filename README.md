# hf-fuyu

The `adept/fuyu-8b` model is broken for `device_map=auto` (e.g. shard layers across gpu).


Edit:  Started writing this and was going to upload to make patching model on other stuff easier but apparently they fixed it and didn't close the gh issue:

https://github.com/huggingface/transformers/pull/29880


Just pushing incase other things need to be fixed and will push model.


# Using package

File paths for the env install:

- `/data/graham/code/hf-fuyu/.venv/lib/python3.11/site-packages/transformers/models/persimmon/modeling_persimmon.py`
  - this is probably where the OOM will occur, e.g. line ~338 in which does the upcast: `attn_weights = nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query_states.dtype)` which you might want to change dtype to float16 (debateable wether this impacts results)
- `/data/graham/code/hf-fuyu/.venv/lib/python3.11/site-packages/transformers/models/fuyu/modeling_fuyu.py`
  - this file contains the `gather_continuous_embeddings` and similar that is what likely is broken in the base model