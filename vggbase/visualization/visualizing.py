"""
The implementation of the visualizer for presenting the samples
and the results.

"""

from typing import List, Optional

import torch

from vggbase.boxes.bbox_convertion import (
    convert_bbox_format,
    convert_model_bbox_format,
)
from vggbase.datasets.data_generic import BaseInputTarget, BaseVGCollatedSamples
from vggbase.models.model_generic import BaseVGModelOutput
from vggbase.learners.learn_generic import BaseVGMatchOutput
from vggbase.visualization.grounding_visualizer import GroundingVisualizer
from vggbase.visualization.base import BaseVisualizer
from vggbase.utils.tensor_utils import BaseNestedTensor
from vggbase.utils.generic_components import BaseVGList


class Visualizer(BaseVisualizer):
    """The visualizer that presents all visible information."""

    def __init__(
        self,
        visualization_path: str,
        is_create_new: bool,
    ):
        super().__init__(visualization_path, is_create_new)
        # Set the ground visualizer
        self.ground_visualizer = GroundingVisualizer(self.visualization_path)
        # Set the mandatory box type
        self.__box_format = "pascal_voc"

    def extract_rgb_images(self, nested_rgbs: BaseNestedTensor, with_pad: bool = False):
        """Extract the rgb images from the nested tensor."""
        if with_pad:
            # Extract the rgb images with pixels padded during
            # creating the nested tensor
            rgb_images = nested_rgbs.tensors.split(1, dim=0)
            rgb_images = [rgb_data.squeeze(0) for rgb_data in rgb_images]
        else:
            # Remove the padded pixels
            rgb_images = nested_rgbs.decompose_via_mask()

        return rgb_images

    def visualize_batch_samples(
        self,
        rgb_images: BaseNestedTensor,
        targets: BaseVGList[BaseInputTarget],
        location: Optional[str] = None,
        visualize_type: str = "samples",
        with_pad: bool = False,
    ):
        """
        Log the raw information of one batch of samples.

        :param rgb_images: One batch of samples,
         of shape, [batch_size, C, H, W]
        :param targets: One batch of targets,
         of shape, len(targets) == batch_size
        :param visualize_type (Optional): A string to show .
        :param batch_idx (Optional): A `Int` presenting the batch idx.
         if provided, samples within this batch will be stored together
         under the folder 'batch_idx/'
        """
        # Extract samples information
        batch_sample_ids = [sample_tg.sample_id for sample_tg in targets]
        batch_phrases = [sample_tg.caption_phrases for sample_tg in targets]
        captions = [sample_tg.caption for sample_tg in targets]

        # Extract the rgb images from the nested tensor
        batch_images = self.extract_rgb_images(rgb_images, with_pad)
        # Convert the image data to the standard for logging
        batch_images = self.get_standard_images(batch_images)

        # Log a batch of samples to disk
        self.ground_visualizer.log_batch_samples(
            batch_names=batch_sample_ids,
            batch_images=batch_images,
            batch_phrases=batch_phrases,
            batch_captions=captions,
            location=location,
            log_type=visualize_type,
        )

    def visualize_batch_bboxes(
        self,
        batch_ids: List[str],
        nested_rgbs: BaseNestedTensor,
        bboxes: List[torch.Tensor],
        bbox_ids: Optional[List[torch.Tensor]] = None,
        bboxes_scores: Optional[List[torch.Tensor]] = None,
        visualize_type: str = "samples",
        location: Optional[str] = None,
        with_pad: bool = False,
        draw_config: dict = None,
    ):
        """Logging bounding boxes with labels on the
        input rgb samples.

        :param nested_rgbs: A `BaseNestedTensor` holding
         one batch rgb samples,
         of shape, [B, C, H, W].
        :param bboxes: A `List` holding bounding boxes of samples,
         each item is a `torch.Tensor` holding bboxes, which shape is
         [N_i, 4], of that sample.
        :param bbox_ids: A `List` holding bboxes labels of samples,
         each item is a `torch.Tensor` holding labels, which shape is
         [N_i], of that sample.
        :param visualize_prefix: A string added to the log file.
            Default, 'outputs'.
        :param location (Optional): location to store the log files.
        :param with_pad: A boolean presenting whether log the bounding boxes
            on the rgb image containing padding pixels. Default, False.
        """
        draw_config = {} if draw_config is None else draw_config

        # Extract the rgb images from the nested tensor
        batch_images = self.extract_rgb_images(nested_rgbs, with_pad)
        # Convert the image data to the standard for logging
        batch_images = self.get_standard_images(batch_images)

        # Convert bboxes to the standard for logging
        bboxes = self.get_standard_data(bboxes)
        bbox_ids = self.get_standard_data(bbox_ids) if bbox_ids is not None else None
        bboxes_scores = (
            self.get_standard_data(bboxes_scores) if bboxes_scores is not None else None
        )
        self.ground_visualizer.log_batch_boxes(
            batch_names=batch_ids,
            batch_images=batch_images,
            batch_boxes=bboxes,
            batch_box_annotations=bboxes_scores,
            location=location,
            file_name="box",
            log_type=visualize_type,
            draw_config=draw_config,
        )

        if bbox_ids is not None:
            bbox_ids = [ids.tolist() for ids in bbox_ids]
            self.ground_visualizer.log_batch_boxes(
                batch_names=batch_ids,
                batch_images=batch_images,
                batch_boxes=bboxes,
                batch_box_ids=bbox_ids,
                batch_box_annotations=bboxes_scores,
                location=location,
                file_name="box_with_ids",
                log_type=visualize_type,
                draw_config=draw_config,
            )

    def visualize_collated_samples(
        self,
        collated_samples: BaseVGCollatedSamples,
        save_location: int = None,
    ):
        """Visualize collated samples."""
        # Convert the box to the format type
        convert_bbox_format(
            BaseVGList([target.vg_bboxes for target in collated_samples.targets]),
            format_type=self.__box_format,
        )

        # Extract the tensor to cpu
        collated_samples.to("cpu")
        targets = collated_samples.targets

        # Log the raw information of one batch of samples
        self.visualize_batch_samples(
            rgb_images=collated_samples.rgb_samples,
            targets=targets,
            visualize_type="samples",
            location=save_location,
        )

        # Log the boxes of one batch of samples
        bboxes = [target.vg_bboxes.bboxes for target in targets]
        bbox_ids = [target.vg_bboxes.bbox_ids for target in targets]

        self.visualize_batch_bboxes(
            batch_ids=[sample_tg.sample_id for sample_tg in targets],
            nested_rgbs=collated_samples.rgb_samples,
            bboxes=bboxes,
            bbox_ids=bbox_ids,
            visualize_type="gt-boxes",
            location=save_location,
            with_pad=False,
        )
        self.visualize_batch_bboxes(
            batch_ids=[sample_tg.sample_id for sample_tg in targets],
            nested_rgbs=collated_samples.rgb_samples,
            bboxes=bboxes,
            bbox_ids=bbox_ids,
            visualize_type="gt-boxes-pad",
            location=save_location,
            with_pad=True,
        )

    def visualize_model_group_outputs(
        self,
        group_idx: int,
        collated_samples: BaseVGCollatedSamples,
        model_outputs: BaseVGModelOutput,
        match_outputs: BaseVGMatchOutput,
        save_location: str = None,
    ):
        """Visualize the model outputs of the `group_idx`."""
        targets = collated_samples.targets
        # Get the bounding boxes
        # of shape, [batch_size, N, 4]
        bboxes = model_outputs.bboxes[:, group_idx, :, :]
        batch_size = bboxes.shape[0]
        # of shape, [batch_size, N, P]
        bboxes_scores = model_outputs.similarity_scores[:, group_idx, :]

        # 1. Plot the raw bounding boxes
        self.visualize_batch_bboxes(
            batch_ids=[sample_tg.sample_id for sample_tg in targets],
            nested_rgbs=collated_samples.rgb_samples,
            bboxes=bboxes,
            bbox_ids=None,
            visualize_type=f"group-{str(group_idx)}-predicted-boxes-pad",
            location=save_location,
            with_pad=True,
        )
        self.visualize_batch_bboxes(
            batch_ids=[sample_tg.sample_id for sample_tg in targets],
            nested_rgbs=collated_samples.rgb_samples,
            bboxes=bboxes,
            bbox_ids=None,
            visualize_type=f"group-{str(group_idx)}-predicted-boxes",
            location=save_location,
            with_pad=False,
        )
        # Get the actual scores of the bounding boxes for each sample
        # of length, [batch_size]
        # of shape for each term, [N, N_i]
        # where N_i is the number of phrases for i-th sample
        bboxes_scores = [
            bboxes_scores[idx][:, : len(targets[idx].caption_phrases)]
            for idx in range(batch_size)
        ]
        # Get the ids of boxes
        # of length, [batch_size]
        # of shape for each term, [N]
        bbox_ids = [bboxes_scores[idx].argmax(1) for idx in range(batch_size)]
        # 2. Plot the raw bounding boxes with idx, thereby the boxes are colored
        # according to the phrase idx
        # Also, the scores of the bounding boxes are predicted ones
        self.visualize_batch_bboxes(
            batch_ids=[sample_tg.sample_id for sample_tg in targets],
            nested_rgbs=collated_samples.rgb_samples,
            bboxes=bboxes,
            bbox_ids=bbox_ids,
            bboxes_scores=bboxes_scores,
            visualize_type=f"group-{str(group_idx)}-predicted-colored-boxes-pad",
            location=save_location,
            with_pad=True,
        )
        self.visualize_batch_bboxes(
            batch_ids=[sample_tg.sample_id for sample_tg in targets],
            nested_rgbs=collated_samples.rgb_samples,
            bboxes=bboxes,
            bbox_ids=bbox_ids,
            bboxes_scores=bboxes_scores,
            visualize_type=f"group-{str(group_idx)}-predicted-colored-boxes",
            location=save_location,
            with_pad=False,
        )
        # 3. Plot the bounding boxes with the scores as the results
        argmax_scores = [
            torch.max(bboxes_scores[idx], dim=0) for idx in range(batch_size)
        ]
        max_scores = [argmax_scores[idx][0] for idx in range(batch_size)]
        max_bboxes = [bboxes[idx][argmax_scores[idx][1]] for idx in range(batch_size)]
        max_bbox_ids = [
            bbox_ids[idx][argmax_scores[idx][1]] for idx in range(batch_size)
        ]

        self.visualize_batch_bboxes(
            batch_ids=[sample_tg.sample_id for sample_tg in targets],
            nested_rgbs=collated_samples.rgb_samples,
            bboxes=max_bboxes,
            bbox_ids=max_bbox_ids,
            bboxes_scores=max_scores,
            visualize_type=f"group-{str(group_idx)}-predicted-result-boxes-pad",
            location=save_location,
            with_pad=True,
        )
        self.visualize_batch_bboxes(
            batch_ids=[sample_tg.sample_id for sample_tg in targets],
            nested_rgbs=collated_samples.rgb_samples,
            bboxes=max_bboxes,
            bbox_ids=max_bbox_ids,
            bboxes_scores=max_scores,
            visualize_type=f"group-{str(group_idx)}-predicted-result-boxes",
            location=save_location,
            with_pad=False,
        )

        # Visualize the matched results
        # This is mainly for the debug purpose to show how the predicted boxes
        # are to be matched with the ground truth
        if match_outputs is not None:
            # Get the matching results
            # of shape, [batch_size, N]
            match_flags = match_outputs.match_bbox_flags[:, group_idx, :]
            # of shape, [batch_size, N]
            match_gt_bbox_indexes = match_outputs.match_gt_bbox_indexes[:, group_idx, :]
            # Output similarity scores
            # of shape, [batch_size, N, P]
            # where P here denotes the number of phrases
            similarity_scores = model_outputs.similarity_scores[:, group_idx, :]

            bboxes = [bboxes[idx][match_flags[idx] > 0] for idx in range(batch_size)]
            # Assign corresponding query indexes to the predicted bounding
            # boxes
            bbox_ids = [
                targets[idx].vg_bboxes.bbox_ids[
                    match_gt_bbox_indexes[idx][match_flags[idx] > 0]
                ]
                for idx in range(batch_size)
            ]
            # Get the similarity scores of the phrase ids
            bbox_scores = [
                similarity_scores[idx][
                    torch.arange(bbox_ids[idx].size(0)), bbox_ids[idx]
                ]
                for idx in range(batch_size)
            ]
            # 4. Plot the bounding boxes that are matched with the ground truth
            # the scores of the bounding boxes are computed by measuring the
            # boxes with the ground truth
            self.visualize_batch_bboxes(
                batch_ids=[sample_tg.sample_id for sample_tg in targets],
                nested_rgbs=collated_samples.rgb_samples,
                bboxes=bboxes,
                bbox_ids=bbox_ids,
                bboxes_scores=bbox_scores,
                visualize_type=f"group-{group_idx}-predicted-gt-matched-boxes-pad",
                location=save_location,
                with_pad=True,
            )

            self.visualize_batch_bboxes(
                batch_ids=[sample_tg.sample_id for sample_tg in targets],
                nested_rgbs=collated_samples.rgb_samples,
                bboxes=bboxes,
                bbox_ids=bbox_ids,
                bboxes_scores=bbox_scores,
                visualize_type=f"group-{group_idx}-predicted-gt-matched-boxes",
                location=save_location,
                with_pad=False,
            )

    def visualize_model_outputs(
        self,
        collated_samples: BaseVGCollatedSamples,
        model_outputs: BaseVGModelOutput,
        match_outputs: BaseVGMatchOutput = None,
        save_location: str = None,
    ):
        """Visualize the model outputs.

        :param model_outputs: The model_outputs here can be the
         original outputs from the model, or the outputs after
         performing matching which leads to BaseVGMatchOutput.
         However, as BaseVGMatchOutput inherits from BaseVGModelOutput,
         we solely put the BaseVGMatchOutput here.
        """

        # Conver the box to the format type
        convert_model_bbox_format(
            model_outputs,
            format_type=self.__box_format,
        )

        model_outputs.to("cpu")
        match_outputs.to("cpu")
        # Visualize the results of groups in one batch
        n_groups = model_outputs.bboxes.shape[1]
        for group_idx in range(n_groups):
            self.visualize_model_group_outputs(
                group_idx=group_idx,
                collated_samples=collated_samples,
                model_outputs=model_outputs,
                match_outputs=match_outputs,
                save_location=save_location,
            )
