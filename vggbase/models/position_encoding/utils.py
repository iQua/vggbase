"""
Useful functions for position encoding

"""

import torch


def generate_1D_position_ids(seq_length, batch_size=1):
    """ Generate 1D position index basd on the given seuence length,
        the position index ranges from 0 to seq_length.

        Args:
            seq_length (int): the required length of the position index.
            batch_size (int): the desired batch size

        Output:
            batch_position_ids (torch.tensor): the generated position ids
                for one batch, thus shape: batch_size, seq_length
                For any two batches with index i and j,
                    batch_position_ids[i] == batch_position_ids[j]

                Within each batch i, the item value is the position id.
    """
    position_ids = torch.arange(start=0, end=seq_length, step=1)

    # convert to batch_size, seq_len
    #   by repeating the position_ids in each batch
    # thus, the position ids in each batch are the same.
    batch_position_ids = position_ids[None, :].repeat(batch_size, 1)

    return batch_position_ids


def generate_2D_position_ids(height, width, batch_size=1, start_pos_index=0):
    """  Generate 1D position index basd on the given height and width,
        For height, the position index ranges from 0 to height - 1.
        For width, the position index ranges from 0 to width - 1.

        Args:
            height (int): the required position length along height direction
                For example: the height size of the image
            width (int): the required position length along height direction
                For example: the height size of the image
            batch_size (int): the required batch size

        Outputs:
            h_batch_position_ids (torch.tensor): a tensor with shape
                batch_size, height, width
                For any two batches with index i and j,
                    batch_position_ids[i] == batch_position_ids[j]

                Within each batch i, the position ids are listed along
                    the column. An example can be viewed in the code below.

            w_batch_position_ids (torch.tensor):  a tensor with shape
                batch_size, height, width
                For any two batches with index i and j,
                    batch_position_ids[i] == batch_position_ids[j]

                Within each batch i, the position ids are listed along
                    the row.  An example can be viewed in the code below.
    """

    h_position_ids = torch.arange(start=start_pos_index,
                                  end=height + start_pos_index,
                                  step=1)
    w_position_ids = torch.arange(start=start_pos_index,
                                  end=width + start_pos_index,
                                  step=1)

    # convert to batch_size, seq_length. For height and width:
    #   For height, h_batch_position_ids: batch_size, height
    #    as the seq_length equals to the height.
    #   For width, w_batch_position_ids: batch_size, width
    #    as the seq_length equals to the width.
    h_batch_position_ids = h_position_ids[None, :].repeat(batch_size, 1)
    w_batch_position_ids = w_position_ids[None, :].repeat(batch_size, 1)

    # convert to the batch_size, height, width
    #  For height, we repeat the column along the width
    #  For width, we repeat the row along the height
    #   Here we want to convert the generated pos ids to strucutres
    #    along the corresponding height and width.
    #   For example, for the pos ids along height, the desired structure
    #    is that these pos ids are placed in the height direction, such as:
    #       [
    #           [[0, 0, 0, 0, 0],
    #            [1, 1, 1, 1, 1],
    #            [2, 2, 2, 2, 2]],
    #           [[0, 0, 0, 0, 0],
    #            [1, 1, 1, 1, 1],
    #            [2, 2, 2, 2, 2]]
    #       ], i.e., the pos ids are listed along the height direction.
    #   For the pos ids along width, the desired structure is:
    #       [
    #           [[0, 1, 2, 3, 4],
    #            [0, 1, 2, 3, 4],
    #            [0, 1, 2, 3, 4]],
    #           [[0, 1, 2, 3, 4],
    #            [0, 1, 2, 3, 4],
    #            [0, 1, 2, 3, 4]]
    #       ], i.e., the pos ids are listed along the width direction.
    # Note: it seems that the torch.meshgrid can also complete this part,
    #   however, how to achieve this with the batch_size axis is not clear to
    #   me. A study is welcomed.
    h_batch_position_ids = h_batch_position_ids[:, :, None].repeat(1, 1, width)
    w_batch_position_ids = w_batch_position_ids[:,
                                                None, :].repeat(1, height, 1)

    return h_batch_position_ids, w_batch_position_ids


def generate_2D_custom_position_ids(batch_size, height, width):
    """ Generate the custom position ids (with specific structure) used in
        our position encoders based on the input height and width.

        Args:
            batch_size (int): batch_size
            height (int): height
            width (int): width

        Outputs:
            custom_pos_ids (torch.tensor): a tensor with shape:
                batch_size, height, width, 2,
                where the pos ids for height and width are stored
                 in the final dim 2.
                Thus,
                 h_position_ids = custom_pos_ids[:, :, :, 0]
                 w_position_ids = custom_pos_ids[:, :, :, 1]
                 with shape batch_size, height, width
    """
    # generate position ids start from index 1
    # h_batch_position_ids: batch_size, height, width
    # likewise for w_batch_position_ids
    h_batch_position_ids, w_batch_position_ids = generate_2D_position_ids(
        height, width, batch_size, start_pos_index=1)

    # add the dim 4 for the further concat
    h_batch_position_ids = h_batch_position_ids[:, :, :, None]
    w_batch_position_ids = w_batch_position_ids[:, :, :, None]
    custom_pos_ids = torch.cat((h_batch_position_ids, w_batch_position_ids),
                               dim=3)

    return custom_pos_ids


def organize_custom_position_ids(h_position_ids, w_position_ids, batch_size=1):
    """ Organize the custom_position_ids (to specific structure) from the given h and w pos ids.

        Args:
            h_position_ids (torch.tensor): a torch tensor with shape:
                height_size or batch_size, height_size
            w_position_ids (torch.tensor): a torch tensor with shape:
                width_size or batch_size, width_size

        Outputs:
            custom_pos_ids (torch.tensor): a tensor with shape:
                batch_size, height, width, 2,
                where the pos ids for height and width are stored
                 in the final dim 2.
                Thus,
                 h_position_ids = custom_pos_ids[:, :, :, 0]
                 w_position_ids = custom_pos_ids[:, :, :, 1]
                 with shape batch_size, height, width
    """
    if h_position_ids.ndim == 1:
        # convert to batch_size, seq_length. For height and width:
        #   For height, h_batch_position_ids: batch_size, height
        #    as the seq_length equals to the height.
        h_batch_position_ids = h_position_ids[None, :].repeat(batch_size, 1)

    if w_position_ids.ndim == 1:
        #   For width, w_batch_position_ids: batch_size, width
        #    as the seq_length equals to the width.
        w_batch_position_ids = w_position_ids[None, :].repeat(batch_size, 1)

    height = h_batch_position_ids.shape[1]
    width = w_batch_position_ids.shape[1]

    # likewise the operations in the func 'generate_custom_position_ids'
    h_batch_position_ids = h_batch_position_ids[:, :, None].repeat(1, 1, width)
    w_batch_position_ids = w_batch_position_ids[:,
                                                None, :].repeat(1, height, 1)

    h_batch_position_ids = h_batch_position_ids[:, :, :, None]
    w_batch_position_ids = w_batch_position_ids[:, :, :, None]
    custom_pos_ids = torch.cat((h_batch_position_ids, w_batch_position_ids),
                               dim=3)

    return custom_pos_ids


def process_1d_inputs(inputs, is_custom_position_ids, device):
    """ Process the inputs for the 1D-based position encoders to obtain the
        further the required item, i.e., batch_position_ids.

        Args:
            inputs: (torch.tensor): a tensor with shape
                    batch_size, seq_length
            is_custom_position_ids: whether the input ids are defined by
                the user.
                    If True, the value of the input tensor is the position idex.
                    Otherwise, the value of the input tensor makes no sense, thus
                        we need to genrate position idex for items.
        Outputs:
            batch_position_ids (torch.tensor): a tensor with shape:
                batch_size, seq_length
    """

    assert inputs.ndim == 2
    inputs.int()

    batch_size, seq_len = inputs.shape

    if is_custom_position_ids:
        batch_position_ids = inputs
    else:
        # generate batch_position_ids
        batch_position_ids = generate_1D_position_ids(seq_length=seq_len,
                                                      batch_size=batch_size)

    batch_position_ids = batch_position_ids.to(device)
    return batch_position_ids


def process_2D_inputs(inputs,
                      is_mask_input,
                      is_custom_position_ids,
                      device=None):
    """ Process the inputs for the 2D-based position encoders to obtain the
        further the required items, i.e., h_batch_position_ids, w_batch_position_ids.

        Args:
            inputs (torch.tensor): a tensor with shape
                    If is_custom_position_ids is True:
                        the expected shape is batch_size, height, width, 2
                        The user defined coords is placed in 2 as [h, w].
                        Check the .utils.py for how to create the required
                        custom_position_ids.
                    else:
                        batch_size, height, width

                    If is_mask_input, the inputs is a tensor containing the
                        mask identifying the positions mask. Then, we need to
                        obtain the position encoding based on the mask.
                        the shape of inputs: batch_size, height, width

            is_mask_input (Boolean): whether the inputs presents the mask used
                for the position encoding, then we need to generate position ids
                based on the mask matrix.

            is_custom_position_ids: whether the input ids are defined by
                the user.
                    If True, the value of the input tensor is the position idex.
                    Otherwise, the value of the input tensor makes no sense, thus
                        we need to genrate position idex for items.
        Outputs:
            h_batch_position_ids (torch.tensor): a tensor with shape
                batch_size, height, width
                For any two batches with index i and j,
                    If not is_custom_position_ids:
                        batch_position_ids[i] == batch_position_ids[j]
                else:
                    the relation between batch_position_ids[i] and
                     batch_position_ids[j] is unclear as they are defined
                     by the user.

                Within each batch i, the position ids are listed along
                    the column.

            w_batch_position_ids (torch.tensor):  a tensor with shape
                batch_size, height, width
                For any two batches with index i and j,
                    If not is_custom_position_ids:
                        batch_position_ids[i] == batch_position_ids[j]
                else:
                    the relation between batch_position_ids[i] and
                     batch_position_ids[j] is unclear as they are defined
                     by the user.

                Within each batch i, the position ids are listed along
                    the row.

    """
    if device is None:
        device = inputs.device

    inputs.int()
    if is_mask_input:
        not_mask = ~inputs
        h_batch_position_ids = not_mask.cumsum(1, dtype=torch.float32)
        w_batch_position_ids = not_mask.cumsum(2, dtype=torch.float32)
    else:
        if is_custom_position_ids:
            assert inputs.ndim == 4
            # the final dim stores the user defined coords.
            assert inputs.shape[-1] == 2
            # obtain the height and width pos ids
            #   shape: batch_size, height, width
            h_batch_position_ids = inputs[:, :, :, 0]
            w_batch_position_ids = inputs[:, :, :, 1]
        else:
            batch_size, height, width = inputs.shape
            h_batch_position_ids, w_batch_position_ids = generate_2D_position_ids(
                height, width, batch_size, start_pos_index=0)
    h_batch_position_ids = h_batch_position_ids.to(device)
    w_batch_position_ids = w_batch_position_ids.to(device)
    return h_batch_position_ids, w_batch_position_ids
