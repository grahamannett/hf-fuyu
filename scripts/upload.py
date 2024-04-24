from huggingface_hub import HfApi

from hf_fuyu.model.modeling_fuyu import FuyuForCausalLM
import argparse


def _make_repo(repo_name: str):
    api = HfApi()
    repo_id = api.create_repo(repo_name, exist_ok=True)
    return repo_id


def upload_model(repo_name):
    model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b")
    model.push_to_hub(repo_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_name", help="Name of the repository", default="besiktas/fuyu-8b"
    )
    args = parser.parse_args()

    upload_model(args.repo_name)
