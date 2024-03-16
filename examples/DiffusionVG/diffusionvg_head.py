"""
The language-guided diffusion head, referred to as LG-DHead.


``LG-DHead`` executes:

0.  receiving, `bboxes_features`, `text_query` and `projected_time` 
    as inputs.
1.  mapping the `text_query` and `projected_time` into the 
    the same space, referred to as `mapped_text` and `mapped_time`.

2. performing the ``LGRCNNBlock`` one by one.

"""

from typing import Type, Optional, List, Tuple, OrderedDict
import logging

from diffusionvg_generic import LGDiffusionVGHeadOutput
from language_guided_rcnn import LGRCNN

import torch
from torch import nn
from torchvision.ops import MultiScaleRoIAlign

from vggbase.models.diffusions import generalized_diffusion_head
from vggbase.config import Config


class LGDiffusionHead(generalized_diffusion_head.GeneralizedDiffusionHead):
    """The head of diffusion model for visual grounding."""

    def __init__(
        self,
        time_embedding_config: Type[Config],
        time_projection_config: Type[Config],
        text_projection_config: Type[Config],
        rois_config: Type[Config],
        box_projection_config: Type[Config],
        head_model_config: Type[Config],
    ) -> None:
        super().__init__(
            time_embedding_config=time_embedding_config,
            time_projection_config=time_projection_config,
            head_model_config=head_model_config,
        )

        backbone_out_channels = rois_config.in_channels

        # set the configurations for ROI (region of interest)
        self.roi_config = rois_config
        # build box pooler.
        self.box_pooler = self.init_roi_pooler(self.roi_config)
        pool_out_size = self.box_pooler.output_size[0]

        in_n_features = backbone_out_channels * pool_out_size**2

        self.text_projector = self.init_text_projector(text_projection_config)
        self.box_projecter = self.init_box_projecter(
            box_projection_config, in_n_features
        )

    def init_time_projector(self, time_projection_config):
        """Initialize the time projector."""
        logging.info("Initialized time projector.")
        in_n_features = time_projection_config.in_features
        hidden_n_features = time_projection_config.hidden_features
        out_n_features = time_projection_config.out_features
        return nn.Sequential(
            nn.Linear(in_n_features, hidden_n_features),
            nn.GELU(),
            nn.Linear(hidden_n_features, out_n_features),
        )

    def init_text_projector(self, text_projection_config):
        """Init the text projector"""
        logging.info("Initialized time projector.")

        in_n_features = text_projection_config.in_features
        out_n_features = text_projection_config.out_features
        return nn.Sequential(
            nn.Linear(in_n_features, out_n_features),
        )

    def init_roi_pooler(self, roi_config):
        """Initialize the roi pooler."""

        feature_layers = roi_config.feature_layers
        pooler_type = roi_config.pooler_type
        pooler_resolution = roi_config.pooler_resolution
        sampling_ratio = roi_config.pooler_sampling_ratio

        box_pooler = MultiScaleRoIAlign(
            featmap_names=feature_layers,
            output_size=pooler_resolution,
            sampling_ratio=sampling_ratio,
        )

        logging.info(
            "Initialized %s with feature layers %s", pooler_type, feature_layers
        )

        return box_pooler

    def init_box_projecter(self, box_projection_config, in_n_features):
        """Initialize the projector for box."""
        logging.info("Initialized box projector.")
        hidden_n_features = box_projection_config.hidden_features
        out_n_features = box_projection_config.out_features
        return nn.Sequential(
            nn.Linear(in_n_features, hidden_n_features),
            nn.GELU(),
            nn.Linear(hidden_n_features, out_n_features),
        )

    def init_head_model_series(self, head_model_config):
        """Initialize the model for this diffusion head."""

        n_repeat = 1
        if hasattr(head_model_config, "n_repeat"):
            n_repeat = head_model_config.n_repeat

        d_model = head_model_config.d_model
        rcnn_config = head_model_config.lgrcnn

        lg_rcnn_head = nn.ModuleList([])
        for _ in range(n_repeat):
            lg_rcnn_head.append(LGRCNN(d_model=d_model, rcnn_config=rcnn_config))

        logging.info("Initialized %d number of `LGRCNNBlock` head", n_repeat)

        return lg_rcnn_head

    def forward(
        self,
        ts_samples: torch.Tensor,
        time_steps: torch.Tensor,
        x_rgbs: OrderedDict[str, torch.Tensor],
        rgbs_hw: List[Tuple[int, int]],
        x_tquery: torch.Tensor,
        tquery_mask: torch.Tensor,
        init_samples_features: Optional[torch.FloatTensor] = None,
    ):
        """
        :param ts_samples: A `torch.Tensor` holding bboxes of the corresponding
         time steps,
         of shape, [bs, N, 4]
         of format, pascal_voc
        :parm time_steps: A `torch.IntTensor` with 1D holding times steps for
         one batch of sample,
         of length, len(time_steps) = bs
        :param x_tquery: tquery: A `torch.FloatTensor` holding the query tokens
         for one batch of samples,
         of shape, [bs, P, D]
         where `D` is the lenght of language embeding features.
         It should have the same shape as input samples.
        :param tquery_mask: (Default) A `torch.BoolTensor` holding the mask
         for the `tquery`,
         of shape, [bs, P].
        :param x_rgbs: A `torch.FloatTensor` holding rgbs data,
         of shape, [bs, C, h, w]
        :param rgbs_hw: A `List` holding the original rgb sizes before
         being padded as nested tensor,
         of length, bs,
         of format, [h, w].
        :param init_samples_features: (Default, None) A `torch.TensorFloat`
         holding initialized features for ``ts_samples``.
        """

        # convert to the shape [bs, 1]
        time_steps = time_steps.reshape(-1, 1)
        # embed t to be one feature space
        # shape, [bs, 1, embed_D_ts]
        time_steps = self.time_embedder(time_steps)
        # convert to [bs, projected_D_ts]
        time_steps = time_steps.squeeze(1)
        time_steps = self.time_projector(time_steps)

        # convert to [bs, P, D_t]
        x_tquery = self.text_projector(x_tquery)

        bboxes = ts_samples
        bs, n_boxes = bboxes.shape[:2]

        series_predicted_scores = []
        series_predicted_bboxes = []

        # obtain bboxes features
        proposal_boxes = list()
        for b_idx in range(bs):
            proposal_boxes.append(bboxes[b_idx])

        # extract rois features
        # of shape, [bs * n_boxes, self.d_model, pooler_resolution[0], pooler_resolution[1]]
        # default `out_channels` of ResNet backbone with FPN is 256
        # we set the core `d_model` to be also 256
        # thus, out_channels here is directly replaced with self.d_model
        rois_features = self.box_pooler(x_rgbs, proposal_boxes, image_shapes=rgbs_hw)
        rois_features = rois_features.view(bs * n_boxes, -1)
        # project the rois features
        # to, [bs, n_boxes, D_b]
        rois_features = self.box_projecter(rois_features).view(bs, n_boxes, -1)

        for _, lg_rcnn_head in enumerate(self.head_model_series):
            vgrnn_outputs = lg_rcnn_head(
                rois_features=rois_features,
                bboxes=bboxes,
                x_tquery=x_tquery,
                tquery_mask=tquery_mask,
                time_steps=time_steps,
            )
            predicted_bboxes = vgrnn_outputs.predicted_bboxes
            if self.return_series_outputs:
                series_predicted_scores.append(vgrnn_outputs.matching_scores)
                series_predicted_bboxes.append(predicted_bboxes)
            else:
                series_predicted_scores = vgrnn_outputs.matching_scores
                series_predicted_bboxes = predicted_bboxes

            bboxes = predicted_bboxes.detach()
            rois_features = vgrnn_outputs.pre_rois_features
            x_tquery = vgrnn_outputs.pre_text_features

        if self.return_series_outputs:
            # obtain the outputs of all series of models
            # shape, [bs, n_repeat, *]
            # where `n_repeat` can be accessed in
            # the function "init_head_model_series".
            series_predicted_scores = torch.stack(series_predicted_scores, dim=1)
            series_predicted_bboxes = torch.stack(series_predicted_bboxes, dim=1)
        else:
            # add to first dimension to make the shape to
            # [*, 1, *]
            series_predicted_scores = series_predicted_scores[:, None, :, :]
            series_predicted_bboxes = series_predicted_bboxes[:, None, :, :]

        return LGDiffusionVGHeadOutput(
            prediction_type=self.prediction_type,
            predictions=series_predicted_bboxes,
            matching_scores=series_predicted_scores,
        )
