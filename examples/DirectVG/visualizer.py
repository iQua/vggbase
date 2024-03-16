"""
Perform visualization to the iterative adjustment of bounding boxes.
"""

import os

import torch
from numpy import random
import networkx as nx
import matplotlib.pyplot as plt


from vggbase.visualization.visualizing import Visualizer
from vggbase.datasets.data_generic import BaseVGCollatedSamples
from vggbase.models.model_generic import BaseVGModelOutput
from vggbase.learners.learn_generic import BaseVGMatchOutput
from vggbase.boxes.bbox_utils import box_iou


def build_edges(bounding_boxes, labels):
    """Build edges for bounding boxes based on their labels"""
    iou_scores, _ = box_iou(bounding_boxes, bounding_boxes)

    edges = set()  # Using a set to avoid duplicates

    # Intra-class connections: Connect each box with its nearest two neighbors of the same label
    for i, label in enumerate(labels):
        same_class_indices = (labels == label).nonzero(as_tuple=True)[0]
        same_class_ious = iou_scores[i][same_class_indices]

        # Zero out the current box's IoU score to avoid self-connection
        same_class_ious[i % len(same_class_ious)] = 0
        if len(same_class_ious) > 2:
            # Exclude the lowest IoUs to keep only the top 2 neighbors
            top2_neighbors = torch.topk(same_class_ious, 2).indices
            for neighbor in top2_neighbors:
                if i != same_class_indices[neighbor]:  # Check to avoid self-connections
                    edges.add(tuple(sorted((i, same_class_indices[neighbor].item()))))

    # Inter-class connections: Randomly select 3-4 boxes from each class to connect to some boxes of other classes
    for class_id in torch.unique(labels):
        class_indices = (labels == class_id).nonzero(as_tuple=True)[0]
        selected_indices = class_indices[
            torch.randperm(len(class_indices))[: random.randint(3, 4)]
        ]

        other_classes_indices = (labels != class_id).nonzero(as_tuple=True)[0]
        for idx in selected_indices:
            # Randomly select 1-2 boxes from other classes to connect
            if len(other_classes_indices) > 0:
                connections = other_classes_indices[
                    torch.randperm(len(other_classes_indices))[: random.randint(1, 2)]
                ]
                for connection in connections:
                    edges.add(tuple(sorted((idx.item(), connection.item()))))

    return list(edges)


class DirectVGVisualizer(Visualizer):
    """A visualizer to present all results of the DirectVG model."""

    def visualize_batch_graphs(
        self, batch_names, batch_boxes, batch_edges, location, log_type
    ):
        """Visualize the graphs of boxes."""
        save_path = self.visualization_path
        save_path = os.path.join(save_path, location)
        for img_idx, image_name in enumerate(batch_names):
            sample_save_path = os.path.join(save_path, image_name, log_type)
            os.makedirs(sample_save_path, exist_ok=True)
            # Create the graphs
            boxes = batch_boxes[img_idx]
            edges = batch_edges[img_idx]
            # Compute the centers of the bounding boxes for plotting purposes
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2

            # Create a mapping of node to its position (center of the corresponding bounding box)
            pos = {i: center.numpy() for i, center in enumerate(centers)}

            G = nx.Graph()
            G.add_edges_from(edges)

            nx.draw(
                G,
                pos,
                with_labels=True,
                node_color="lightblue",
                edge_color="gray",
                node_size=100,
                font_weight="bold",
                font_size=8,
            )
            plt.axis("off")  # Remove axes and labels
            plt.savefig(
                os.path.join(sample_save_path, "graphs.png"),
                format="png",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.savefig(
                os.path.join(sample_save_path, "graphs.pdf"),
                format="pdf",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.savefig(
                os.path.join(sample_save_path, "graphs.svg"),
                format="svg",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

    def visualize_iter_graph(
        self,
        i_iter,
        collated_samples,
        model_outputs,
        match_outputs,
        save_location,
    ):
        """Visualize the i_iter-th iterative adjustment of bounding boxes."""
        batch_size = model_outputs.bboxes.shape[0]
        # Get the bboxes of the current iteration
        # of length, batch_size
        # of shape for each term, [N, 4]
        iter_bboxes = [model_outputs.bboxes[idx, i_iter] for idx in range(batch_size)]
        # Get the similarity scores of the current iteration
        # of length, batch_size
        # of shape for each term, [N, P]
        similarities = [
            model_outputs.similarity_scores[idx, i_iter] for idx in range(batch_size)
        ]

        batch_argmax_idxs = [torch.argmax(sim, dim=1) for sim in similarities]
        batch_box_tgt_ids = [tgt.vg_bboxes.bbox_ids for tgt in collated_samples.targets]

        batch_box_ids = [
            tgt_ids[argmax_idxs]
            for tgt_ids, argmax_idxs in zip(batch_box_tgt_ids, batch_argmax_idxs)
        ]

        # Plot the graph
        batch_edges = [
            build_edges(bounding_boxes=boxes, labels=boxes_id)
            for boxes, boxes_id in zip(iter_bboxes, batch_box_ids)
        ]

        self.visualize_batch_graphs(
            batch_names=[sample_tg.sample_id for sample_tg in collated_samples.targets],
            batch_boxes=iter_bboxes,
            batch_edges=batch_edges,
            location=save_location,
            log_type=f"iter-{i_iter}-graph",
        )

    def visualize_model_outputs(
        self,
        collated_samples: BaseVGCollatedSamples,
        model_outputs: BaseVGModelOutput,
        match_outputs: BaseVGMatchOutput = None,
        save_location: str = None,
    ):
        # Plot the inherent results of VGGbase
        super().visualize_model_outputs(
            collated_samples=collated_samples,
            model_outputs=model_outputs,
            match_outputs=match_outputs,
            save_location=save_location,
        )

        num_iters = model_outputs.bboxes.shape[1]

        for i_iter in range(num_iters):
            self.visualize_iter_graph(
                i_iter,
                collated_samples,
                model_outputs,
                match_outputs,
                save_location,
            )
