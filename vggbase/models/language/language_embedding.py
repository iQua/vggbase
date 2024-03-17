"""
The language module to encode the text by using the vector embedding.

It mainly supports the bert or other pretrained model to obtain the
 embedding.
"""

from typing import Type

import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer
from transformers import logging
from vggbase.config import Config

logging.set_verbosity_warning()


class LanguageEmbeddingModule(nn.Module):
    """
    A standard text embedding architecture based on the pre-trained model.
    """

    def __init__(self, language_module_config: Type[Config]):
        super().__init__()
        language_model_name = language_module_config.model_name
        pretrained_model_dir = language_module_config.model_path
        self.pooling_type = language_module_config.pooling_type

        self.language_model = SentenceTransformer(
            model_name_or_path=language_model_name,
            cache_folder=pretrained_model_dir,
        )

        self.n_encoded_features = language_module_config.features

    def freeze_parameters(self):
        """Freeze parameters of the language embedder."""
        for param in self.language_model.parameters():
            param.requires_grad = False

    def mean_pooling(self, text_embeddings, attention_masks):
        """Mean Pooling - Take attention mask into account for correct averaging

        Args:
            text_embeddings (torch.tensor): a torch with shape
                batch_size, number_of_embeddings, embedding_dim
            attention_masks (torch.tensor): a torch with shape
                batch_size, number_of_embeddings

        Returns:
            mean_pooled_embeddings (torch.tensor): a torch with shape
                batch_size, embedding_dim
        """

        input_mask_expanded = (
            attention_masks.unsqueeze(-1).expand(text_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(text_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled_embeddings = sum_embeddings / sum_mask
        return mean_pooled_embeddings

    def forward(
        self, nested_word_tokens_id: torch.IntTensor, word_masks: torch.BoolTensor
    ):
        """Forward the token ids for embedding.

        :param nested_word_tokens_id: A `torch.IntTensor` holding words id,
         of shape, [batch_size, P, L]
        :param word_masks: A `torch.boolTensor` holding the words mask,
         of shape, [batch_size, P, F]

        :return phrases_embedding: A `torch.FloatTensor` holding the embed
         features for phrases,
         of shape, [batch_size, P, D]
         where `D` is features length.
        """

        # as the mask in nested tensor follows the format that
        #   True: masked; False: non-masked
        # thus need to convert the format of that in transformers
        #   1: non-masked; 0: masked
        word_masks = ~word_masks
        bs, P = nested_word_tokens_id.shape[:2]
        flatten_nested_tensors = nested_word_tokens_id.view(bs * P, -1)
        flatten_mask = word_masks.view(bs * P, -1)

        encoded_batch_phrases = {
            "input_ids": flatten_nested_tensors,
            "attention_mask": flatten_mask,
        }

        self.language_model(encoded_batch_phrases)

        # sentence_embedding, of shape [bs * P, n_features]
        # token_embeddings, of shape [bs * P, L, n_features]
        phrases_embedding = encoded_batch_phrases["sentence_embedding"]
        token_embeddings = encoded_batch_phrases["token_embeddings"]

        if self.pooling_type == "mean":
            token_embeddings = self.mean_pooling(
                text_embeddings=token_embeddings, attention_masks=flatten_mask
            )
        phrases_embedding = phrases_embedding.view(bs, P, -1)

        return phrases_embedding
