"""
A model to generate bounding boxes randomly.
"""

import asyncio
import concurrent.futures

import torch
import torch.nn as nn
import numpy as np

from vggbase.boxes import bbox_extension
from vggbase.models.model_generic import BaseVGModelInput
from vggbase.models.model_generic import BaseVGModelOutput


def generate_phrase_bboxes(n_boxes: int, n_phrases: int):
    """Generate boxes for each phrase."""
    # Generate boxes with shape
    # [n_boxes, 4], [ctr_x, ctr_y, width, height]
    # and the label with shape [n_boxes,]
    return bbox_extension.generate_random_bboxes(
        n_bboxes=n_boxes, device=None
    ), torch.rand(size=(n_boxes, n_phrases))


class RandomVG(nn.Module):
    """The VG algorithm to generate bounding boxes randomly."""

    def __init__(
        self,
        n_proposals: int,
    ):
        super().__init__()
        # Set how many proposals to generate
        self.n_proposals = n_proposals

    async def generate_bboxes(self, n_boxes_per_phrase: int, n_phrases: int):
        """Generate boxes for one batch of samples."""
        loop = asyncio.get_event_loop()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            tasks = [
                loop.run_in_executor(
                    executor, generate_phrase_bboxes, n_boxes, n_phrases
                )
                for n_boxes in n_boxes_per_phrase
            ]
            generated_bboxes = await asyncio.gather(*tasks)

        # Get boxes with shape [n_proposals, 4]
        return torch.cat([bboxes[0] for bboxes in generated_bboxes], dim=0), torch.cat(
            [bboxes[1] for bboxes in generated_bboxes], dim=0
        )

    async def generate_batch_bboxes(self, n_boxes_per_phrase, batch_size, n_phrases):
        """Generate boxes for one batch of samples."""
        batch_bboxes = await asyncio.gather(
            *[
                self.generate_bboxes(n_boxes_per_phrase, n_phrases)
                for _ in range(batch_size)
            ]
        )

        # Get the batch of boxes with shape [batch_size, n_proposals, 4]
        # of format, [ctr_x, ctr_y, width, height]
        # Get the batch of box scores with shape [batch_size, n_proposals, n_phrases]
        return torch.stack([bboxes[0] for bboxes in batch_bboxes], dim=0), torch.stack(
            [bboxes[1] for bboxes in batch_bboxes], dim=0
        )

    def forward(self, inputs: BaseVGModelInput):
        """Foward the model to generate bounding boxes randomly."""
        # Get how many boxes to generate for each phrase
        batch_size, n_phrases = inputs.text_samples.shape[:2]
        pieces = np.array_split(np.arange(0, self.n_proposals), n_phrases)
        n_boxes_per_phrase = [len(piece) for piece in pieces]

        # Generate bounding boxes randomly
        loop = asyncio.get_event_loop()
        # Get boxes with shape [batch_size, n_proposals, 4]
        # of format, [ctr_x, ctr_y, width, height]
        batch_boxes, similarity_scores = loop.run_until_complete(
            self.generate_batch_bboxes(n_boxes_per_phrase, batch_size, n_phrases)
        )

        # Extend to the target shape
        # [batch_size, 1, n_proposals, 4]
        # [batch_size, 1, n_proposals, n_phrases]
        batch_boxes = batch_boxes[:, None, :, :]
        similarity_scores = similarity_scores[:, None, :, :]
        board_hws = torch.tensor(
            [target.vg_bboxes.board_hw for target in inputs.targets]
        )
        device = inputs.rgb_samples.device
        batch_boxes = batch_boxes.to(device)
        similarity_scores = similarity_scores.to(device)
        board_hws = board_hws.to(device)
        return BaseVGModelOutput(
            bboxes=batch_boxes,
            similarity_scores=similarity_scores,
            bbox_type="yolo",
            board_hws=board_hws,
        )
