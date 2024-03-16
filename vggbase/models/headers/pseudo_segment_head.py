"""
The visual grounding heads for obtaining the pseudo segmentation map.

This is the pseudo segmentation map header, which obtains the segmentation
    map based on the pyramid attention map obtained by the visual grounding model.

It receives the input whose type is a list, and each item is a attention map with shape:
    batch_size, num_queries, attn_h, attn_w

Note, the pyramid stores the attention map following the rule that the attn map of the final
 layer is stored as the first item.

We insist that the attn maps in pyramid are extracted from the successive layers.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PseudoSegGroundingHeader(nn.Module):
    """ This is the grounding header to obtain the segmentation map
        for corresponding queries.

        It is named as Pseudo because:
            - this is not actually the general segmentaion map used
                in the image segmentation task.
            - it does not utilize the general deconv but use the
                interpolate (bilinear) to extend the map to the original input size.

        Thus, no trainable weights in this seg header.

        """

    def __init__(self):
        """ Initializes the header.

            Nothing is needed to be initialized here.
        """

        super().__init__()
        # import pdb;pdb.set_trace()

    def forward(self, attn_maps_pyramid, target_upsample_scale, vaild_size):
        """ Merging the attention maps pyramid following the mechnism of the FCN network.

            Args:
                attn_maps_pyramid (list): holds the attention map obtained
                        by each layer of the grounding model. Each item:
                        batch_size, num_queries, attn_h, attn_w
                        Order of attn maps:
                            final - start
                target_upsample_scale (int): holds the target upsample scale
                            that maps the pyramid bottom to.
                            For example, h, w if the pyramid bottom.
                                the target size is:
                                    h * target_upsample_scale, w * target_upsample_scale


                vaild_size (list): holds the size of attn containing the
                        vaild values,
                        for example, removing the padding terms
                # target_size (list): the final size that wants to upsample
                #         the attn map to.

            Output:
                merged_attn_map (torch.tensor): the merged attention map whose
                    shape is:
                    batch_size, num_queries, attn_h, attn_w

                    where attn_h, attn_w is the size of attn_maps_pyramid[0],
                    i.e. the pyramid bottom
        """

        debug_level = 1

        merged_attn_map = None

        # visit the cross layer from final to the start
        #  as we need to perfrom the merging from final to the start
        #  i.e., -> mean bilinearly upsampling achieved by nn.functional.interpolate
        #    layer 3: 1/32 -> 1/16
        #                      |  +=  1/16 -> 1/8
        #    layer 2:         1/16             | += 1/8 -> 1/4
        #    layer 1:                         1/8           |  += 1/4
        #    layer 0:                                       1/4
        # Note here 1/4 is the final target size as the input image
        #   is partitioned into patches and each patch size is 4.
        # Of course, if input patch_size is not 4, we the target is
        #   1/patch_size
        for attn_idx, attn_map in enumerate(attn_maps_pyramid):

            if attn_idx == 0:
                # convert attn_map to [0, 1]
                merged_attn_map = attn_map
            else:

                cur_h, cur_w = merged_attn_map.shape[-2:]
                to_attn_h, to_attn_w = attn_map.shape[-2:]
                scale_factor = (cur_h // to_attn_h, cur_w // to_attn_w)

                # first need to upsample the scores to the size of current layer
                merged_attn_map = F.interpolate(merged_attn_map,
                                                scale_factor=scale_factor,
                                                mode='bilinear',
                                                align_corners=True)
                merged_attn_map += attn_map

        # Convert the currentmerged_unnorm_scores to the start size

        merged_attn_map = F.interpolate(merged_attn_map,
                                        scale_factor=target_upsample_scale,
                                        mode='bilinear',
                                        align_corners=True)

        # obtain the atten map that is consistent with the input
        #   batch_size, num_queries, Ph + (#padding for patches merging part)*scale, Pw + (#padding for patches merging part)*scale
        # convert to [0, oo]
        merged_attn_map = F.relu(merged_attn_map)
        # to [0, 1]
        merged_attn_map = torch.tanh(merged_attn_map)

        # For example, In the boottransvg
        # Note that the map size of upsampled merged_attn_map may not be consistent with
        #   the src_Ph, src_Pw,
        #  The main reason is that in each layer, the x is padded within the patch merging
        #   for (H+#pad)%2==0. The padding is performed along the bottom of h
        #   and along the right of width
        #   Thus, we can just remove the padding part from the upsampled map by:
        vaild_h, vaild_w = vaild_size
        merged_attn_map = merged_attn_map[:, :, :vaild_h, :vaild_w]

        return merged_attn_map