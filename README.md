# hf-fuyu

The `adept/fuyu-8b` model is broken for `device_map=auto` (e.g. shard layers across gpu).


Edit:  Started writing this and was going to upload to make patching model on other stuff easier but apparently they fixed it and didn't close the gh issue:

https://github.com/huggingface/transformers/pull/29880


Just pushing incase other things need to be fixed and will push model.