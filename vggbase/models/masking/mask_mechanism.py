"""
The implementations of visual masking methods for
masking out some regions from the visual input.

Square masking, originally inspired by the paper
"Context Encoders: Feature Learning by Inpainting".

Grid masking, analyzed in work SimMIM.

Block masking, originally inspired by the paper
"Beit: Bert pre-training of image transformers".

Block GMM masking, implemented by us.
A novel way to achieve block masking.

Random masking, originally inspired by the paper
"Masked autoencoders are scalable vision learners" with
official code:
https://github.com/huggingface/transformersmodels/vit_mae/modeling_vit_mae.py.

For the comparsion of different masking mechanisms, please access the paper titled
"SimMIM: a Simple Framework for Masked Image Modeling".


Note, Block GMM masking is the new one proposed by us.
"""

from typing import Optional, List
import random
import math

import torch
import torch.distributions as D
import numpy as np


def square_masking(sequence, seq_root_size, mask_ratio: float = 0.5):
    """
    Perform square masking by masking out a square area around the
    center of the input 1D or 2D tensor.

    :param sequence: A `torch.LongTensor` of shape `(batch_size, sequence_length, num_features)`.
    :param seq_root_size: A `list` of containing `int`, which denotes the length of each dimension.
      For 2D sequence, sequence_length = list[0] * list[1]
      For 1D sequence, sequence_length = list[0]
    :param mask_ratio: A `float` to the proporation required to be masked out.

    :return sequence_masked: A `torch.LongTensor` of shape `(batch_size, len_keep, num_features)`.
        where len_keep = int(sequence_length * (1 - mask_ratio))
    :return mask: A `torch.Float32Tensor` of shape `(batch_size, sequence_length)`.
        1, masked, 0: unmasked
    :return seq_ids: A `torch.Int64Tensor` of shape `(batch_size, sequence_length)` denoting
        the id of each item in the sequence.
    """
    batch_size, seq_length, num_features = sequence.shape
    seq_ids = torch.arange(0, seq_length, device=sequence.device).repeat(batch_size, 1)

    # compute the length of the square
    len_keep = int(seq_length * (1 - mask_ratio))
    square_len = np.ceil(np.sqrt(len_keep))
    len_keep = square_len * square_len

    # get the center of the sequence,
    # center_ids shape:
    # if 1D, [seq_length],
    # if 2D, [height, width]
    center_ids = torch.Tensor(seq_root_size, device=sequence.device) // 2
    center_ids = center_ids.int()

    # obtain the
    # - left corner where the square starts
    # - right corner where the square ends
    half_sqeuare_len = int(square_len // 2)
    start_left_corner = center_ids - half_sqeuare_len
    end_right_corner = center_ids + half_sqeuare_len

    # mask the square subset
    # mask shape:
    # if 1D, [square_len]
    # if 2D, [height, width]
    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.zeros(seq_root_size, device=sequence.device)
    if len(seq_root_size) == 1:
        mask[start_left_corner:end_right_corner] = 1
    else:
        mask[
            start_left_corner[0] : end_right_corner[0],
            start_left_corner[1] : end_right_corner[1],
        ] = 1

    # convert mask to 1D
    mask = mask.flatten()

    # copy the square subet to all samples of the batch
    ids_keep = torch.nonzero(mask).flatten()
    ids_keep = ids_keep.repeat(batch_size, 1).type(torch.int64)

    # obtained the sequence after being masked
    # [batch_size, len_keep, num_features_length]
    sequence_masked = torch.gather(
        sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, num_features)
    )

    # repeat mask for the whole batch
    # [batch_size, len_keep]
    mask = mask.repeat(batch_size, 1)

    return sequence_masked, mask, seq_ids


def grid_masking(
    sequence,
    seq_root_size,
    mask_rect_intervals: List[int],
    mask_rect_sizes: List[int],
    mask_ratio: float = 0.5,
):
    """Perform grid masking by masking out rectangles of one sequence.

    This masking mechanism is initially proposed in work [1].

    [1]. Chen et.al, GridMask Data Augmentation.

    :param sequence: A `torch.LongTensor` of shape `(batch_size, sequence_length, num_features)`.
    :param seq_root_size: A `list` of containing `int`, which denotes the length of each dimension.
      For 2D sequence, sequence_length = list[0] * list[1]
      For 1D sequence, sequence_length = list[0]
    :param mask_rect_intervals: A `list` denoting distance between two adjacent mask rectangles.
      Each item holds the distance of the corresponding dimension.
    :param mask_rect_sizes: A `list` denoting size of the mask rectangle.
      Each item holds the length of rectangle in that dimension.
    :param mask_ratio: A `float` to the proporation required to be masked out.
    """
    batch_size, seq_length, num_features = sequence.shape
    root_dims = len(seq_root_size)
    len_keep = int(seq_length * (1 - mask_ratio))
    len_mask = seq_length - len_keep
    seq_ids = torch.arange(0, seq_length, device=sequence.device).repeat(batch_size, 1)

    dim_rects_position = list()
    for dim_idx in range(root_dims):
        # obtain the mask info of one dim
        dim_length = seq_root_size[dim_idx]
        rect_length = mask_rect_sizes[dim_idx]
        half_rect_length = rect_length // 2
        rect_interval = mask_rect_intervals[dim_idx]

        num_intervals = (dim_length - rect_length) // rect_interval
        left_sapce = dim_length - num_intervals * rect_interval - rect_length

        # compute the start/end positions of the range to place
        # masks
        mask_start_pos = left_sapce
        mask_end_pos = dim_length - left_sapce
        # generate the center of rectangle along this dimension
        dim_rects_center = torch.arange(mask_start_pos, mask_end_pos, rect_interval)
        # compute the upper left and bottom right corner of this dimension
        dim_rects = torch.stack(
            [
                torch.arange(center - half_rect_length, center + half_rect_length)
                for center in dim_rects_center
            ]
        ).flatten()

        dim_rects_position.append(dim_rects)

    # generate the mask ids based on the rectangle positions of all dimensions
    mask_ids = torch.meshgrid(dim_rects_position)
    # stack the ids to be [height_ids, width_ids, 2]
    # convert to be [height_ids * width_ids, 2]
    # stack the mask_ids to be [num_masks, 2]
    mask_ids = torch.stack(mask_ids, dim=2).view((-1, 2))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.zeros(seq_root_size, device=sequence.device)
    # assign masked positions to be 1
    mask[mask_ids[:, 0], mask_ids[:, 1]] = 1

    # convert shape of mask
    # from seq_root_size
    # to be 1D sequence, [seq_length]
    mask = mask.flatten()

    # get position ids of masked parts
    ids_keep = torch.nonzero(mask == 0)[:, 0]
    ids_keep = ids_keep.repeat(batch_size, 1)

    sequence_masked = torch.gather(
        sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, num_features)
    )

    # extend mask to the whole batch
    mask = mask.repeat(batch_size, 1)

    return sequence_masked, mask, seq_ids


def blockwise_masking(
    sequence,
    seq_root_size,
    mask_ratio: float = 0.5,
    min_num_seq: int = 4,
    max_num_seq: Optional[int] = None,
    min_aspect: int = 0.3,
    max_aspect: Optional[int] = None,
):
    """
    Perform block masking by visiting all positions repeatly.

    This is the original implementation of blockwise masking of:
        Block masking, originally inspired by the paper
        "Beit: Bert pre-training of image transformers".

    Note, currently, this only supports the 2D mode.

    :param sequence: A `torch.LongTensor` of shape `(batch_size, sequence_length, num_features)`
    :param seq_root_size: A `list` of containing `int`, which denotes the length of each dimension.
      For 2D sequence, sequence_length = list[0] * list[1]
      For 1D sequence, sequence_length = list[0]
    :param mask_ratio: A `float` to the proporation required to be masked out.
    :param min_num_seq: A `int` denoting minimum number of sequence within one block.
    :param max_num_seq: A `int` denoting maximum number of sequence within one block.
    :param min_aspect: A `int` denoting minimum aspect of one block.
    :param max_aspect: A `int` denoting maximum aspect of one block.
    """
    batch_size, seq_length, num_features = sequence.shape
    height, width = seq_root_size
    len_keep = int(seq_length * (1 - mask_ratio))
    len_mask = seq_length - len_keep
    seq_ids = torch.arange(0, seq_length, device=sequence.device).repeat(batch_size, 1)

    max_num_seq = len_mask if max_num_seq is None else max_num_seq
    max_aspect = max_aspect or 1 / min_aspect
    log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def _mask(mask, max_mask_seq):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(min_num_seq, max_mask_seq)
            aspect_ratio = math.exp(random.uniform(*log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < width and h < height:
                top = random.randint(0, height - h)
                left = random.randint(0, width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_seq:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    mask = torch.zeros(size=seq_root_size, device=sequence.device)
    mask_count = 0
    while mask_count < len_mask:
        max_mask_seq = len_mask - mask_count
        max_mask_seq = min(max_mask_seq, max_num_seq)

        delta = _mask(mask, max_mask_seq)
        if delta == 0:
            break
        else:
            mask_count += delta

    # convert the mask shape,
    # from [height, width]
    # to [seq_length], where seq_length = height * width
    mask = mask.flatten()

    # obtain position ids of un-masked parts
    ids_keep = torch.nonzero(mask == 0)[:, 0]
    # convert to [batch_size, len_keep]
    ids_keep = ids_keep.repeat(batch_size, 1)
    # obtain the masked sequence
    # [batch_size, len_keep]
    sequence_masked = torch.gather(
        sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, num_features)
    )

    # extend mask to the whole batch
    mask = mask.repeat(batch_size, 1)
    return sequence_masked, mask, seq_ids


def blockwise_gmm_masking(
    sequence, seq_root_size, mask_ratio: float = 0.5, num_gaussians: int = 3
):
    """
    Perform block masking by relying on the Gaussian Mixture Models (GMM).

    :param sequence: A `torch.LongTensor` of shape `(batch_size, sequence_length, num_features)`
    :param seq_root_size: A `list` of containing `int`, which denotes the length of each dimension.
      For 2D sequence, sequence_length = list[0] * list[1]
      For 1D sequence, sequence_length = list[0]
    :param mask_ratio: A `float` to the proporation required to be masked out.
    :param num_gaussians: A `int` denoting how many gaussians in the GMM.
    """
    batch_size, seq_length, num_features = sequence.shape
    num_dim = len(seq_root_size)
    len_keep = int(seq_length * (1 - mask_ratio))
    len_mask = seq_length - len_keep
    seq_ids = torch.arange(0, seq_length, device=sequence.device).repeat(batch_size, 1)

    # construct Gaussian Mixture Modle consisting of `num_gaussians` equally
    # weighted bivariate normal distributions
    mix = D.Categorical(
        torch.ones(
            num_gaussians,
        )
    )

    # generate means and variances for each component
    # in the GMM
    means = torch.ones(num_gaussians, num_dim, device=sequence.device)
    vars = torch.ones(num_gaussians, num_dim, device=sequence.device)
    for dim_idx in range(num_dim):
        half_size = seq_root_size[dim_idx] // 2
        quarter_size = half_size // 2
        means[:, dim_idx] = torch.randint(
            quarter_size, half_size + quarter_size, size=(num_gaussians,)
        )
        vars[:, dim_idx] = torch.randint(1, half_size, size=(num_gaussians,))

    # create GMM
    comp = D.Independent(D.Normal(means, vars), 1)
    gmm = D.mixture_same_family.MixtureSameFamily(mix, comp)

    # obtain the position idxes of the sequence
    # seq_item_pos should be:
    # if 1D sequence, 1D positions
    # if 2D sequence, height position and width position
    full_mask = torch.ones(seq_root_size, device=sequence.device)
    seq_item_pos = torch.nonzero(full_mask)
    # obtain the likelihood of each position
    points = seq_item_pos.repeat(batch_size, 1, 1)
    points_logprob = gmm.log_prob(points)
    points_expprob = torch.exp(points_logprob)
    points_prob = points_expprob / torch.sum(points_expprob)
    # sample number of positions based on the likelihood as
    # the to-be-masked ones
    ids_mask = torch.multinomial(points_prob, len_mask)

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.zeros([batch_size, seq_length], device=sequence.device)
    for bs in range(batch_size):
        mask[bs, ids_mask[bs]] = 1

    ids_keep = torch.nonzero(mask == 0)[:, 1].split(len_keep)
    ids_keep = torch.stack(ids_keep)

    sequence_masked = torch.gather(
        sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, num_features)
    )

    return sequence_masked, mask, seq_ids


def random_masking(sequence, mask_ratio: float = 0.5, noise: Optional[float] = None):
    """
    Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
    noise.

    :param sequence: A `torch.LongTensor` of shape `(batch_size, sequence_length, num_features)`
    :param mask_ratio: A `float` to the proporation required to be masked out.
    :param noise: A `torch.FloatTensor` of shape `(batch_size, sequence_length)`
        *optional*)  which is mainly used for testing purposes to control
        randomness and maintain the reproducibility

    :return sequence_masked: A `torch.LongTensor` of shape `(batch_size, len_keep, num_features)`
    :return mask: A `torch.Float32Tensor` of shape `(batch_size, sequence_length)`
    :return ids_restore: A `torch.Int64Tensor` of shape `(batch_size, sequence_length)`
    """
    batch_size, seq_length, num_features = sequence.shape
    len_keep = int(seq_length * (1 - mask_ratio))

    # create random noise in [0, 1]
    if noise is None:
        noise = torch.rand(batch_size, seq_length, device=sequence.device)

    # sort noise for each sample
    # ascend: small is keep, large is remove
    # ids_shuffle: [batch_size, sequence_length]
    #  holding the shuffled ids
    # ids_restore: [batch_size, sequence_length]
    #  holding the original position of shuffled
    #  ids.
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    # ids_keep: [batch_size, len_keep]
    ids_keep = ids_shuffle[:, :len_keep]

    # extract the masked sequence
    # sequence_masked: [batch_size, len_keep, num_features]
    sequence_masked = torch.gather(
        sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, num_features)
    )

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([batch_size, seq_length], device=sequence.device)
    mask[:, :len_keep] = 0

    # unshuffle to get the binary mask
    # mask: [batch_size, sequence_length]
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return sequence_masked, mask, ids_restore

