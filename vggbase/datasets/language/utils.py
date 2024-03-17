"""
The useful functions used for language process part.

"""
from typing import List


def find_sub_list(to_search_list: List[int], target_list: List[int]):
    """Find the sub list `to_search_list` in the list target_list.

    :param to_search_list: A `List` that are required to be
     found.
    :param target_list (list): The target `List` where the 
     `to_search_list` is to be found.


    :return : A `List` containing two items [a1, a2], which
     denotes the start and ene postion of `to_search_list`
     in `target_list`, i.e., target_list[a1: a2] == to_search_list.

     e.g.: to_search_list: [1, 2, 4]
        target_list: [0, 4, 1, 2, 4, 6, 8]
        search_pos: [2, 4]

    """
    sll = len(to_search_list)
    search_pos = list()
    for ind in (i for i, e in enumerate(target_list) if e == to_search_list[0]):
        if target_list[ind : ind + sll] == to_search_list:
            search_pos = [ind, ind + sll]
            return search_pos

    return search_pos


def get_largest_phrase_number(target_captions_phrases):
    """Obtain the largest number of phrases in different captions."""

    largest_number = max(
        [
            len(caption_phrases)
            for _, caption_phrases in enumerate(target_captions_phrases)
        ]
    )

    return largest_number


def get_longest_phrase_length(target_captions_phrases: List[List[str]], tokenizer):
    """Get the max length sub list in the target list.

    
    :param target_captions_phrases (list): a 2-depth nested list.

    :param tokenizer: A defined tokenizer used to
     process the text.

        For example:
            [
                [ 'Military personnel', 'greenish gray uniforms', 'matching hats' ],
                [ 'a man', 'in red', 'boat with a women' ],
                [ 'a man', 'a women' ],
            ].
            we can obtain 6 as 'boat with a women' contains maximum 4 words including
                the start [TOKEN] and end [TOKEN].

        Note: For the bert tokenizer, a noun word and its plural are different.
            Exp, 'oranges', the tokenzier encodes it as 5925, 1116
                 'orange', the tokenzier encodes it as 5925

        Therefore, the length of phrase 'oranges on left' is 6
                    the length of phrase 'orange on left' is 5
    """
    # flatten the nested list to be one list, in which
    #  each item is a list containing the phrase string
    flatten_captions_phrases = [
        ph for cap_phrases in target_captions_phrases for ph in cap_phrases
    ]

    token_ids = tokenizer(flatten_captions_phrases, padding=False)["token_type_ids"]

    max_phrase_length = max(len(token_ids[idx]) for idx in range(len(token_ids)))

    return max_phrase_length
