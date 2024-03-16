"""
The visual grounding heads for obtaining the bounding boxes.

This is the general box regression header, which predicts the bounding
 boxes based on the output (for example, relation or multi-modal
 representation) obtained by the visual grounding model.

It receives the input with shape:
    batch_size, num_total_predictions, dim

0- Initiall, we need to convert the input's shape to
 batch_size, dim, num_total_predictions
 and then to
 batch_size, dim, num_total_predictions, 1
 for convolutional computation.

1- It first maps the dim to the head_reduced_dim, based on the convolutional
 computation method for efficiency.
2- It then utilizes the MLP to obtain the prediction coordinates:
    batch_size, num_total_predictions, 4
    where the 4 denotes: [center_x, center_y, width, height]

"""

import torch.nn as nn

from ..module_utils import MLP


class DirectBoxRegGroundingHeader(nn.Module):
    """ This is the direct grounding header to predict the bounding boxes
        coordinates for corresponding queries.

        It is named as DirectBoxReg because:
            - uses the box regression to predict the bounding boxes.
            - predicts the (center_x, center_y, height, width).
                These values are normalized in [0, 1], relative to
                the size of each individual image (disregarding possible padding).
        """

    def __init__(self,
                 grounding_output_features,
                 head_reduced_features,
                 grouped_norm_channel=32,
                 num_mlp_layers=3):
        """ Initializes the header.

            Inputs:
                grounding_output_dim (int): the output #features of the grounding
                    model. It can be obtained by applying:
                        ground_model.num_channels[-1]
                head_reduced_features (int): the #features for the grounding head.
                grouped_norm_channel (int): number of groups to separate the
                    channels into.
        """

        super().__init__()
        # import pdb;pdb.set_trace()

        hidden_features = grounding_output_features
        self.input_proj = nn.Sequential(
            nn.Conv2d(hidden_features, head_reduced_features, kernel_size=1),
            nn.GroupNorm(grouped_norm_channel, head_reduced_features),
        )
        # in this MLP layer, the hidden dim of the intermediate layers
        #   is the same, i.e., head_reduced_features
        # 4 here for coordinates
        self.bbox_embed = MLP(head_reduced_features, head_reduced_features, 4,
                              num_mlp_layers)

        ## init
        nn.init.xavier_uniform_(self.input_proj[0].weight, gain=1)
        nn.init.constant_(self.input_proj[0].bias, 0)
        ##

    def forward(self, det):
        """ Predict the direct bounding boxes coordinates.

            :param det (torch.tensor): the obtained det tokens from
                grounding model, shape:
                    batch_size, dim, num_total_predictions

            :return outputs_coord: A tensor for predicted bounding box coords,
                    [batch_size, num_total_predictions, 4]
        """

        # a linear projection by perfomring the conv2d
        # as Linear can perform a Conv2d as a special case
        # with det, [B, dim, num_total_predictions]
        # add a new dimention to the final
        # after unsqueeze, det, [B, dim, num_total_predictions, 1]
        # then perform conv2d to obtain,
        # det, [B, head_reduced_features, num_total_predictions, 1]
        # remove the final dimension to obtain,
        # det, [B, head_reduced_features, num_total_predictions]
        # permute to general structure for embedding
        # after permute, det, [B, num_total_predictions, head_reduced_features]
        x = self.input_proj(det.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)

        # predictions
        outputs_coord = self.bbox_embed(x).sigmoid()

        return outputs_coord