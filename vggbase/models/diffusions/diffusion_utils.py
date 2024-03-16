"""
Implementations of useful utilities for
diffusion models.

"""

from functools import partial

import torch


def extract_items(
    source_items: torch.Tensor, extract_ids: torch.Tensor, out_n_dims: int
):
    """Extract values with extract_ids from the source_items,
    then outputing the values by expanding to target n_dims."""
    n_items, *_ = extract_ids.shape
    out = source_items.gather(-1, extract_ids)
    # extract target items with shape
    # [n_items, *]
    return out.reshape(n_items, *((1,) * (out_n_dims - 1)))


def identity(inputs, *args, **kwargs):
    """Identity function."""
    return inputs


def process_samples_fn(**kwargs):
    """Process the samples when possible."""
    if "is_clip" in kwargs:
        return partial(torch.clamp, min=-1.0, max=1.0)
    else:
        return identity


# normalization functions
def normalize_to_neg_one_to_one(imgs):
    return imgs * 2 - 1


def unnormalize_to_zero_to_one(imgs):
    return (imgs + 1) * 0.5
