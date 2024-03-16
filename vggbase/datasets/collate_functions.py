"""
Implementation of Collate functions to process the loaded samples.
"""

from typing import List

import torch

from vggbase.datasets.data_generic import (
    BaseVGSample,
    BaseVGCollatedSamples,
    BaseInputTarget,
)
from vggbase.datasets.language import LanguageDynamicTokenizer
from vggbase.boxes.bbox_generic import BaseVGBBoxes
from vggbase.utils.tensor_utils import BaseNestedTensor, DynamicMaskNestedTensor
from vggbase.utils.generic_components import BaseVGList
from vggbase.utils.tensor_utils import (
    nested_2d_tensor_from_list,
    nested_3d_tensor_from_list,
)


class FormatSamplesCreator:
    """A creator used to generate one batch of format samples from the loaded raw samples."""

    def __init__(self, language_tokenizer: LanguageDynamicTokenizer):
        # Set the language_tokenizer
        self.language_tokenizer = language_tokenizer

    def collate_function(self, batch_samples: List[BaseVGSample]):
        """
        Create the format samples from one batch of loaded samples.

        :return BaseVGCollatedSamples: A format collated sample required by
         the subsequent learning.
        """

        (
            padded_word_tokens_ids,
            word_tokens_masks,
            phrase_token_mask,
        ) = self.language_tokenizer.forward(
            [vg_sample.caption_phrases for vg_sample in batch_samples]
        )

        # Create nested rgbs data
        # of shape, [batch_size, C, H, W]
        # with mask, of shape [batch_size, H, W]
        nested_rgbs = BaseNestedTensor(
            *nested_3d_tensor_from_list(
                [vg_sample.image_data for vg_sample in batch_samples]
            )
        )
        # Create nested text data
        # of shape, [batch_size, P, L]
        # with mask, of shape [batch_size, P, L]
        nested_texts = DynamicMaskNestedTensor(
            *nested_2d_tensor_from_list(
                tensor_list=padded_word_tokens_ids, external_masks=word_tokens_masks
            ),
            mask_p=phrase_token_mask
        )

        # Create the targets by visiting one batch
        # of input samples
        targets = BaseVGList([])
        for vg_sample in batch_samples:
            targets.append(
                BaseInputTarget(
                    sample_id=vg_sample.sample_id,
                    caption=vg_sample.caption,
                    caption_phrases=vg_sample.caption_phrases,
                    vg_bboxes=BaseVGBBoxes(
                        bboxes=torch.FloatTensor(
                            vg_sample.caption_phrases_bboxes.bboxes
                        ),
                        labels=torch.IntTensor(
                            vg_sample.caption_phrases_bboxes.bboxes_label
                        ),
                        bbox_ids=torch.IntTensor(
                            vg_sample.caption_phrases_bboxes.bbox_ids
                        ),
                        board_hw=tuple(vg_sample.caption_phrases_bboxes.board_hw),
                        bbox_type=vg_sample.caption_phrases_bboxes.bbox_type,
                    ),
                )
            )

        return BaseVGCollatedSamples(
            rgb_samples=nested_rgbs, text_samples=nested_texts, targets=targets
        )
