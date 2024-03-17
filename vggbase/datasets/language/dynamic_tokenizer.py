"""
The language dynamic padding is used to pad the phrases/sentences/text in one batch to
 be the consistent length.

We call it 'dynamic' because there is no need to set the max length. It performs on
 one batch of samples, thus only need to pad the samples in current batch to be
 consistent length.

The tokenizer of the transformer (https://huggingface.co/docs/transformers/preprocessing)
 can be used directly to achieve the dynamic padding as we only need to set hyper-parameters.

The special tokens are:
    [CLS]: Added to the start of the text string
    [SEP]: Added to the end of the text string

Setting all hyper-parameters to False to achieve the dynamic padding.

Once the name of one mask contain *attention*, its masking format should 
follow: 1 means masked 0 means unmasked.
"""

from typing import List, Dict, Type

import torch

from transformers import AutoTokenizer
from transformers import logging
from .utils import get_longest_phrase_length, get_largest_phrase_number

from vggbase.config import Config

logging.set_verbosity_warning()


class LanguageDynamicTokenizer(object):
    r"""Implemente the dynamic tokenizer for the caption/phrases/text.
        The main purpose is to pad and encode the text data.
    Args:
     tokenizer_name (str): the name of the tokenizer to be used.
      This is supported by the tokenizer from
      transformer, i.e., pretrained_model_name_or_path
     tokenizer_path (str): same as the 'cache_dir' in transformer
      tokenizer.
     max_caption_length (int): used to define the maximum words of
      captions. Default False to support the dynamic padding, i.e.,
      padding the captions in one batch with the longest caption as
      the benchmark.

     max_caption_n_phrases (int): the maximum number of
      phrases in one caption. This one is used to truncate
      the caption containing too many phrases.
      Default False or None to support the dynamic padding, i.e.,
      within one batch, padding the number of phrases in different
      captions to be the same as the caption that contains the
      maximum phrases.
     max_phrase_length (int): the maximum number of words in one phrase.

    Note: all maximum length here includes the start and end tokens! Thus, if you set the
        'max_caption_length' to 2, the encoded caption will only contain the start and end
        token. In 'bert' tokenzier, they are 101 and 102.
    """

    def __init__(
        self,
        language_config: dict,
    ):
        tokenizer_name = language_config["tokenizer_name"]
        self.max_caption_length = language_config["max_caption_length"]
        self.max_caption_n_phrases = language_config["max_caption_n_phrases"]
        self.max_phrase_length = language_config["max_phrase_length"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_name,
            force_download=False,
        )
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id

    def pad_captions(self, captions: List[List]) -> Dict[str, torch.Tensor]:
        """Padding the captions that are presented as a nested list.

        :param captions: A `Nest List`, in which each item is a list that
         contains the string of the phrase
         For example:
            [
                ['Military personnel greenish gray uniforms matching hats'],
                ['a man in red is standing in boat with a women']
            ].

        :return: A `Dict` containing three items:
         input_ids, token_type_ids,ttention_mask.

         If self.max_caption_length is True:
          input_ids: num_captions (or batch_size) x max_caption_length
          attention_mask: num_captions (or batch_size) x max_caption_length
         else:
          input_ids: num_captions (or batch_size) x max_length_in_batch
          attention_mask: num_captions (or batch_size) x max_length_in_batch
        """
        # flatten the captions into a list in which each item is a caption string
        #  to: ['Military personnel greenish gray uniforms matching hats',
        #       'a man in red is standing in boat with a women']
        flatten_captions = [caption for caption in captions]

        # pad to the longest sequence in the batch
        if self.max_caption_length is None:
            encoded_batch_captions = self.tokenizer(
                flatten_captions,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
            )

        else:
            encoded_batch_captions = self.tokenizer(
                flatten_captions,
                padding="max_length",
                max_length=self.max_caption_length,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
            )

        return encoded_batch_captions

    def align_captions_phrases_in_number(self, captions_phrases):
        """Padding the phrases for one batch of captions, making each caption contain
        the same number of phrases.

        :param captions_phrases: A `nested List` containing the phrases for a batch of
         captions.
            For example,
            [
                [ 'Military personnel', 'greenish gray uniforms', 'matching hats' ],
                [ 'a man', 'in red', 'boat with a women' ],
                [ 'a man', 'a women']
            ].

        :return aligneded_captions_phrases: A `nested List` containing the aligned phrases
         for all captions, i.e., each caption has same number of phrases.
            For example,
            [
                [ 'Military personnel', 'greenish gray uniforms', 'matching hats' ],
                [ 'a man', 'in red', 'boat with a women' ],
                [ 'a man', 'a women', '[PAD]' ]
            ].
        :return phrases_align_attention_mask: A `torch.Tensor` holding the masks for aligned phrases.
         of shape, [batch_size, max_number_phrases],
         where `max_number_phrases` depends on whether the language_padder is defined to
         pad a batch of captions to be the `max_caption_n_phrases` or to the maximum
         number of phrases in this batch.
            For example:
            [
                [ 1,   1,   1 ],
                [ 1,   1,   1 ],
                [ 1,   1,   0 ]
            ].

        """
        largest_phrase_number = get_largest_phrase_number(captions_phrases)

        # pad the number of phrases in captions to tbe the largest number
        #  in one batch.
        aligned_captions_phrases = list()
        phrases_align_attention_mask = list()

        def pad_caption_phrases(caption_phrases, to_number_of_phrases):
            """Pad the caption phrases to the desired number.

            :param caption_phrases: A `List` holding phrases of one caption.
             For example: [ 'a man', 'a women' ]
            :param to_number_of_phrases: A `Int` denoting the desired number
             of phrases after padding.

            :return padded_caption_phrases: A `List` holding the padded phrases
             of one caption.
             For example: [ 'a man', 'a women', '[PAD]' ]
            :return padded_attention_mask: A `List[int]` holding the attention
             mask for padding phrases of this caption.
             For example: [ 1, 1, 0 ]
             1 means unmasked, 0 means masked.
             len(padded_attention_mask) = len(padded_caption_phrases)
            """
            padded_caption_phrases = []
            padded_attention_mask = []
            for ph_idx in range(to_number_of_phrases):
                if ph_idx < len(caption_phrases):
                    padded_caption_phrases.append(caption_phrases[ph_idx])
                    padded_attention_mask.append(1)
                else:
                    padded_caption_phrases.append(self.pad_token)
                    padded_attention_mask.append(0)

            return padded_caption_phrases, torch.as_tensor(padded_attention_mask)

        if self.max_caption_n_phrases is None:
            for caption_phrases in captions_phrases:
                padded_caption_phrases, mask = pad_caption_phrases(
                    caption_phrases, largest_phrase_number
                )
                aligned_captions_phrases.append(padded_caption_phrases)
                phrases_align_attention_mask.append(mask)
        else:
            for caption_phrases in captions_phrases:
                padded_caption_phrases, mask = pad_caption_phrases(
                    caption_phrases, self.max_caption_n_phrases
                )
                aligned_captions_phrases.append(padded_caption_phrases)
                phrases_align_attention_mask.append(mask)

        return aligned_captions_phrases, torch.stack(phrases_align_attention_mask)

    def pad_captions_phrases_words(self, captions_phrases):
        """Padding the phrases to have the same number of words.
        These are two-levels padding requirements.
         - First, different captions have different number of phrases.
            But, if you perform ``align_captions_phrases_in_number`` beforehead,
            then only the padding below is performed.
         - Second, within each caption, phrases contain various words. Thus, padding
            each phrase in the caption to tbe same length.

        :param captions_phrases: A `nested List`, in which each item is also a list
         that contains the phrases of the corresponding caption
            For example:
            [
                [ 'Military personnel', 'greenish gray uniforms', 'matching hats' ],
                [ 'a man', 'in red', 'boat with a women' ],
                [ 'a man', 'a women' or [ 'a man', 'a women', '[PAD]' ]
            ].

        :return word_padded_captions_phrases: A `nested List` nested list that is the same
         structure of the input `captions_phrases`.
         But each item is a dict containing `input_ids`, `token_type_ids`, `attention_mask`
         for the correponding caption's phrases.
         where the size of the `input_ids` is [number_of_phrases, padding_length].
         where the `padding_length` is either the `max_phrase_length` or the longest
         phrase length in this batch.
         where `attention_mask` is a `torch.IntTensor` holding the mask for the
         corresponding caption, of shape [number_of_phrases, padding_length]

         len(word_padded_captions_phrases) == len(captions_phrases)
        """
        # count the phrases information in the batch of captions
        # captions_phrases_count = [len(phrases) for phrases in captions_phrases]

        longest_phrase_length = get_longest_phrase_length(
            captions_phrases, self.tokenizer
        )

        word_padded_captions_phrases = list()

        for cap_phrases in captions_phrases:
            # flatten the cap_phrases, thus each item is a string
            flatten_cap_phrases = [ph for ph in cap_phrases]
            if self.max_phrase_length is None:
                padded_caption_phrases = self.tokenizer(
                    flatten_cap_phrases,
                    padding="max_length",
                    max_length=longest_phrase_length,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
            else:
                padded_caption_phrases = self.tokenizer(
                    flatten_cap_phrases,
                    padding="max_length",
                    max_length=self.max_phrase_length,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=True,
                )

            word_padded_captions_phrases.append(padded_caption_phrases)

        return word_padded_captions_phrases

    def forward(self, text_queries):
        """Forward the text queries to padded ones contained by the nested tensor.

        :param text_queries: A `nested List` containing text queries,
         Each item is a list containing the queries for corresponding caption.
         - For referitgame dataset,
            [
                [ 'monitor on left' ],
                [ 'guy sitting down wearing red shirt' ]
            ]
         - For F30KE dataset,
            [
                [ 'Three girls', 'a crowd', 'plastic containers'   ],
                [ 'A child', 'a ladder', 'another child', 'a rope' ],
            ].

        :return word_tokens_ids: A `torch.IntTensor` containing the word tokens's id
         of shape [batch_size, num_phrases, num_words].
        :return word_tokens_masks: A `torch.BoolTensor` containing the masks for word tokens
         of shape [batch_size, num_phrases, num_words].
        :return phrase_tokens_masks: A `torch.BoolTensor` containing the mask for phrase tokens,
         of shape [batch_size, num_phrases].
        """
        (
            aligned_captions_phrases,
            aligned_attention_mask,
        ) = self.align_captions_phrases_in_number(captions_phrases=text_queries)
        word_padded_captions_phrases = self.pad_captions_phrases_words(
            captions_phrases=aligned_captions_phrases
        )

        word_tokens_ids = []
        word_attention_masks = []
        for item in word_padded_captions_phrases:
            word_tokens_ids.append(item["input_ids"])
            word_attention_masks.append(item["attention_mask"])

        word_tokens_ids = torch.stack(word_tokens_ids)
        # obtain the `word_tokens_masks` who follows the vggbase style
        # True means masked while False means unmasked.
        word_tokens_masks = torch.stack(
            [~word_mask.bool() for word_mask in word_attention_masks]
        )
        phrase_tokens_masks = ~aligned_attention_mask.bool()

        return word_tokens_ids, word_tokens_masks, phrase_tokens_masks


if __name__ == "__main__":
    captions = [
        ["Military personnel greenish gray uniforms matching hats"],
        ["a man in red is standing in boat with a women"],
        ["a man and a women sleep in a beautiful wooden bed with golves"],
        [
            "a man and a women stand in front of a man and women while other men and women are playing"
        ],
    ]

    captions_phrases = [
        [["Military personnel"], ["greenish gray uniforms"], ["matching hats"]],
        [["a man"], ["in red"], ["boat with a women"]],
        [["a man"], ["a women"], ["a beautiful wooden bed with golves"]],
        [["a man"], ["a women"], ["a man"], ["women"], ["other men and women"]],
    ]

    from collections import namedtuple

    # Declaring namedtuple()
    Language_config = namedtuple(
        "Language_config",
        [
            "tokenizer_name",
            "tokenizer_path",
            "max_caption_length",
            "max_caption_n_phrases",
            "max_phrase_length",
        ],
    )

    language_padder = LanguageDynamicTokenizer(
        Language_config("bert", "data/", 10, False, False)
    )
    encoded_captions = language_padder.pad_captions(captions=captions)

    language_padder = LanguageDynamicTokenizer(
        Language_config("bert", "data/", 10, False, 3)
    )
    encoded_captions = language_padder.pad_captions_phrases(
        captions_phrases=captions_phrases
    )
