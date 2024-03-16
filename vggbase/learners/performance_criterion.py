"""
The implementations of the performance criterion for visual grounding model methods.

The idea of the code for this part comes from two parts:

    1. For the bounding boxes evaluation part, our code depends on the git repo
    https://github.com/rafaelpadilla/review_object_detection_metrics.
    2. coco evaluation part with the it repo
    https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py

"""

from typing import List, Optional
import logging

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops.boxes import box_convert

from vggbase.evaluators.base import BaseEvaluation
from vggbase.datasets.data_generic import BaseInputTarget
from vggbase.models.model_generic import BaseVGModelOutput
from vggbase.utils.generic_components import BaseVGList
from vggbase.learners.matcher import HungarianMatcher
from vggbase.config import Config


class PerformanceCriterion(BaseEvaluation):
    """This class computes the performance for visual grounding methods."""

    def __init__(self, matcher: HungarianMatcher, eval_config: Config):
        """Create the criterion for performance.

        The criteria for the visual grounding are different from those of
        object detection and other label-relied tasks. The main reason is
        that each prediction should correspond to one specific query.
        Then, the predictions can be regarded as the response to the input queries.

        In this criterion, such an alignment between the prediction and the query is
        denoted by the 'predicted_bboxes_label'.

        :param matcher: A module able to compute a matching between targets and proposals.
        :param iou_thresholds_values: See the 'MeanAveragePrecision' below for details.
        :param rec_thresholds_values: See the 'MeanAveragePrecision' below for details.
        :param max_queries_thresholds: See the 'MeanAveragePrecision' below for details.

        """

        super().__init__(eval_config)

        self.matcher = matcher

        self.iou_thresholds_values = self.performance_config.box.iou_thresholds_values
        self.rec_thresholds_values = self.performance_config.box.rec_thresholds_values
        self.max_queries_thresholds = self.performance_config.box.max_queries_thresholds

        self.similarity_threshold = self.performance_config.similarity.prob_threshold

        self.bbox_metric = None
        self.segm_metric = None

        self.metrics = None

    def reset_metrics(self):
        """Define the two metrics"""
        self.bbox_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=self.iou_thresholds_values,
            rec_thresholds=self.rec_thresholds_values,
            max_detection_thresholds=self.max_queries_thresholds,
            class_metrics=False,
        )

        self.segm_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="segm",
            iou_thresholds=self.iou_thresholds_values,
            rec_thresholds=self.rec_thresholds_values,
            max_detection_thresholds=self.max_queries_thresholds,
            class_metrics=False,
        )

    def show_metrics(self, epoch, batch_idx, global_step: Optional[int] = None):
        """Present the obtained metrics on the screen."""
        self.compute_metrics()
        delimiter = ", "
        if global_step is not None:
            header = f" Performance: [{epoch}]/[{batch_idx}]/[{global_step}]"
        else:
            header = "  Performance: [{epoch}]/[{batch_idx}]]"

        log_msg = delimiter.join([header, "{meters}"])

        loss_str = []
        for name, meter in self.metrics.items():
            loss_str.append(f"{name}: {str(meter)}")
        logging.info(log_msg.format(meters=delimiter.join(loss_str)))

    def compute_metrics(self):
        """Compute the metrics.

        These metrics work with DDP in PyTorch and PyTorch Lightning by default.
        When .compute() is called in distributed mode, the internal state of
        each metric is synced and reduced across each process, so that the
        logic present in .compute() is applied to state
        information from all processes.
        """
        metrics = self.bbox_metric.compute()
        self.metrics = {k: v.tolist() for k, v in metrics.items()}

    def organize_metric_format(
        self,
        matched_bboxes: List[torch.FloatTensor],
        matched_bboxes_labels: List[torch.IntTensor],
        matched_bboxes_scores: List[torch.FloatTensor],
        batch_gt_boxes: List[torch.Tensor],
        batch_gt_labels: List[torch.Tensor],
    ):
        """Organzie the obtained results to be the metric format.

        :param matched_bboxes: A `List` containing bboxes that are matched with the
         ground truth (gt) bboxes,
         of length, len(matched_bboxes) == batch size
         of format, each item is a `torch.FloatTensor` with shape [N_i, 4] whose
          coordinate is 'xyxy'
        :param matched_bboxes_labels: A `List` containing labels of matched_bboxes based
         on the similarity scores,
         of length, len(matched_bboxes) == batch size
         of format, each item is a `torch.IntTensor` with shape [N_i]
        :param matched_bboxes_scores: A `List` containing scores of matched_bboxes from
         the similarity scores,
         of same structure as the `matched_bboxes_labels`
        :param batch_gt_boxes: A `List` containing ground truth bboxes for queries
            of one batch of samples.
        :param batch_gt_labels: A `List` containing ground truth labels for queries
            of one batch of samples.

        Note, for all these lists, each item is a tensor corresponding to the specific sample.

        The format should be:
            preds = [
                dict(
                    boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0]]),
                    scores=torch.tensor([0.536]),
                    labels=torch.tensor([0]),
                    )
                ]
            target = [
                dict(
                    boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0]]),
                    labels=torch.tensor([0]),
                    )
                ]
        where each item in the list is a dict that corresponds to a single image.
        """

        preds = [
            {
                "boxes": bboxes,
                "labels": labels,
                "scores": scores,
            }
            for bboxes, labels, scores in zip(
                matched_bboxes, matched_bboxes_labels, matched_bboxes_scores
            )
        ]

        target = [
            {"boxes": gt_boxes, "labels": gt_labels}
            for gt_boxes, gt_labels in zip(batch_gt_boxes, batch_gt_labels)
        ]

        return preds, target

    @torch.no_grad()
    def forward(
        self,
        model_outputs: BaseVGModelOutput,
        samples_target: BaseVGList[BaseInputTarget],
    ):
        """Forward the module to obtain the performance."""

        eval_inputs = self.make_compatible(
            model_outputs=model_outputs, samples_target=samples_target
        )
        targets = eval_inputs.samples_target_eval
        outputs = eval_inputs.model_outputs_eval

        matching_indices = self.matcher(
            model_outputs=model_outputs,
            samples_target=samples_target,
        )

        # each predicted bounding box will correspond to
        # one target bounding box.
        # [bs, n_groups, N, 4]
        src_bboxes = outputs.predicted_vg_bboxes.bboxes
        batch_size, _, _ = src_bboxes.shape[:3]
        # to [bs, n_groups * N, 4]
        src_bboxes = src_bboxes.reshape(batch_size, -1, 4)
        # of shape, [bs_src_n_bboxes, 4]
        # where `bs_src_n_bboxes = \sum_{i=1}^{bs} sample_matched_idxs[i]`
        # where sample_matched_idxs = indices[0]
        matched_src_bboxes = [
            src_bboxes[bs_idx][src_idx]
            for bs_idx, (src_idx, _) in enumerate(matching_indices)
        ]

        target_boxes = [
            tgt.vg_bboxes.bboxes[tgt_bbox_idsx]
            for tgt, (_, tgt_bbox_idsx) in zip(targets, matching_indices)
        ]

        batch_target_ids = [tgt.vg_bboxes.bbox_ids for tgt in targets]
        batch_target_labels = [tgt.vg_bboxes.labels for tgt in targets]

        target_labels = [
            tgt.vg_bboxes.labels[tgt_bbox_idsx]
            for tgt, (_, tgt_bbox_idsx) in zip(targets, matching_indices)
        ]

        # [bs, n_groups, N, P]
        matching_scores = outputs.predicted_vg_bboxes.similarity_scores
        best_match_labels, best_match_scores = self.obtain_best_matching_labels(
            matching_scores=matching_scores,
            matching_indices=matching_indices,
            target_bbox_ids=batch_target_ids,
            target_bboxes_label=batch_target_labels,
        )
        # convert desired case `xyxy`
        matched_src_bboxes = [
            box_convert(src_bboxes, in_fmt=self.eval_box_coord, out_fmt="xyxy")
            for src_bboxes in matched_src_bboxes
        ]
        target_boxes = [
            box_convert(bboxes, in_fmt=self.eval_box_coord, out_fmt="xyxy")
            for bboxes in target_boxes
        ]

        assert (
            len(matched_src_bboxes)
            == len(best_match_labels)
            == len(best_match_scores)
            == len(target_boxes)
            == len(target_labels)
        )

        preds, target = self.organize_metric_format(
            matched_src_bboxes,
            best_match_labels,
            best_match_scores,
            target_boxes,
            target_labels,
        )

        self.bbox_metric.update(preds, target)
