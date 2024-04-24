import torch
from transformers import FuyuForCausalLM as BaseFuyuForCausalLM


class FuyuCombineEmbeddingsMixin:
    """
    layer to combine embeddings in models that do not already allow for multimodal inputs
    """

    def combine_embeddings(
        self,
        word_embeddings: torch.Tensor,
        continuous_embeddings: list[torch.Tensor]
        | torch.Tensor,  #  allow a tensor containing of images to be passed
        image_patch_input_indices: torch.Tensor,
    ):
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/fuyu/modeling_fuyu.py#L168

        for batch_idx in range(word_embeddings.shape[0]):
            dst_indices = torch.nonzero(
                image_patch_input_indices[batch_idx] >= 0, as_tuple=True
            )[0]
            src_indices = image_patch_input_indices[batch_idx][dst_indices]
            # Check if we have more indices than embeddings. Note that we could have fewer indices if images got truncated.
            if src_indices.shape[0] > continuous_embeddings[batch_idx].shape[0]:
                src_indices = src_indices[: continuous_embeddings[batch_idx].shape[0]]
                dst_indices = dst_indices[: len(src_indices)]

            word_embeddings[batch_idx][dst_indices] = continuous_embeddings[
                batch_idx
            ].to(word_embeddings.device)[src_indices]

            # output_embeddings[batch_idx, dst_indices] = continuous_embeddings[batch_idx].to(src_indices.device)[src_indices]
        return word_embeddings


class FuyuForCausalLM(FuyuCombineEmbeddingsMixin, BaseFuyuForCausalLM):
    pass
