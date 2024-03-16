"""
The visual grounding heads for obtaining the bounding boxes.

This is the window-based box regression header, which predicts the bounding
 boxes based on the output of each window obtained by the visual grounding model.

It receives the input with shape:
    batch_size, num_query_windows, num_queries, dim


0- Initiall, we need to convert the input's shape to
 batch_size, num_query_windows * num_queries, dim
 and then to
 batch_size, dim, num_query_windows * num_queries
 finally to
 batch_size, dim, num_query_windows * num_queries, 1
 thus the convolutional computation can be used.

1- It first maps the dim to the head_reduced_dim, based on the convolution
 computation method for efficiency.
2- It then utilizes the MLP to obtain the prediction coordinates:
    batch_size, num_queries, 4
    where the 4 denotes: [center_x, center_y, width, height]

"""

import torch.nn as nn

from ..module_utils import MLP


class DirectWindowRegGroundingHeader(nn.Module):
    """ This is the direct grounding header to predict the bounding boxes
        coordinates for corresponding queries.

        It is named as DirectBoxReg because:
            - uses the box regression to predict the bounding boxes.
            - predicts the (center_x, center_y, height, width).
                These values are normalized in [0, 1], relative to
                the size of each individual image (disregarding possible padding).
        """
    def __init__(self,
                 grounding_output_dim,
                 head_reduced_dim,
                 grouped_norm_channel=32,
                 num_mlp_layers=3):
        """ Initializes the header.

            Inputs:
                grounding_output_dim (int): the output dim of the grounding
                    model. It can be obtained by applying:
                        ground_model.num_channels[-1]
                head_reduced_dim (int): the channel dim for the grounding head.
                grouped_norm_channel (int): number of groups to separate the
                    channels into.
        """

        super().__init__()
        # import pdb;pdb.set_trace()

        hidden_dim = grounding_output_dim
        self.input_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, head_reduced_dim, kernel_size=1),
            nn.GroupNorm(grouped_norm_channel, head_reduced_dim),
        )
        # in this MLP layer, the hidden dim of the intermediate layers
        #   is the same, i.e., head_reduced_dim
        # 4 here for coordinates
        self.bbox_embed = MLP(head_reduced_dim, head_reduced_dim, 4,
                              num_mlp_layers)

        ## init
        nn.init.xavier_uniform_(self.input_proj[0].weight, gain=1)
        nn.init.constant_(self.input_proj[0].bias, 0)
        ##

    def forward(self, det):
        """ Predict the direct bounding boxes coordinates.

            det (torch.tensor): the obtained det tokens from
                grounding model, shape:
                    batch_size, dim, det_token_num

            outputs_coord (torch.tensor): the predicted bounding box coords,
                    shape, batch_size, det_token_num, 4
        """

        # projection
        # obtain x, batch_size, det_token_num, head_reduced_dim
        x = self.input_proj(det.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)

        # predictions
        outputs_coord = self.bbox_embed(x).sigmoid()

        return outputs_coord