"""
Implementation of the bboxes structure, which formats the bounding boxes into 
different structures, such as graphs. 
"""

import dgl
from typing import List, Tuple

import networkx as nx
from scipy.spatial import cKDTree

from vggbase.boxes.bbox_generic import BaseVGBBoxes
from vggbase.boxes.bbox_convertion import convert_bbox_format


class GroundingGraph:
    """
    A base class to represent bounding boxes as the graph in which
    each node corresponds to a bounding box in the image.
    We should know that the node id should be aligned with the index of the corresponding bounding box in the input bboxes.
    """

    def __init__(self):
        """
        Initialize the node with the bounding boxes.
        """
        # Default type of the bounding boxes
        self.__box_type = "pascal_voc"

    def create_bbox_edges(self, vg_bboxes: BaseVGBBoxes):
        """
        Create the edges for the bounding boxes which have been
        converted to 'self.__box_type' format.
        """

        # Get the bboxes
        bboxes = vg_bboxes.bboxes.numpy()

        # Assuming boxes_tensor is your tensor of boxes
        centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2

        # Use scipy.spatial.cKDTree for efficient nearest neighbor search
        tree = cKDTree(centers)

        # Query the two nearest neighbors for each point, excluding the point itself
        _, nearest_neighbors_indices = tree.query(centers, k=3)

        # The first column is the point itself, so we use the second and third columns for the two nearest neighbors
        # Get a 2D numpy array of the two nearest neighbors for each box
        # Each row corresponds to a box, and the two columns are the indices of the two nearest neighbors
        return nearest_neighbors_indices[:, 1:3]

    def build_nx_graph(
        self, vg_bboxes: BaseVGBBoxes, edges: List[Tuple[int, int]] = None
    ):
        """
        Insert the bounding boxes into the networkx graph.
        """
        # Create the undirected graph
        nx_graph = nx.MultiGraph()

        # Conver the bboxes to the desired format
        convert_bbox_format([vg_bboxes], format_type=self.__box_type)

        # Create edges
        if edges is None:
            # Use the default edges in which each node is connected to its two nearest neighbors
            edges = self.create_bbox_edges(vg_bboxes)

        # Get [N_i, 4] bboxes
        bboxes = vg_bboxes.bboxes.detach().numpy()
        # Get [N_i] labels and bbox_ids
        labels = bboxes.labels.detach().numpy()
        bbox_ids = bboxes.bbox_ids.detach().numpy()

        for box_idx, box in enumerate(bboxes):

            # Add node
            node_id = f"B-{box_idx}"
            nx_graph.add_node(
                node_id, bbox=box, label=labels[box_idx], bbox_id=bbox_ids[box_idx]
            )
            # Add edges
            neighbors = edges[box_idx]
            nx_graph.add_edges_from([(box_idx, neighbor) for neighbor in neighbors])

        return nx_graph

    def build_graph(self, bboxes: BaseVGBBoxes):
        """
        Build the grounding graph of DLG from the graph of the networkx.

        :param bboxes: A `BaseVGBBoxes` holding the bounding boxes.
        """
        # Build the networkx graph
        nx_graph = self.build_nx_graph(bboxes)

        # Create a graph from the networkx
        dlg_graph = dgl.from_networkx(nx_graph)
        # Add the bbox coordinates to the graph
        dlg_graph.ndata["coordinates"] = bboxes.bboxes
        dlg_graph.ndata["labels"] = bboxes.labels
        dlg_graph.ndata["bbox_ids"] = bboxes.bbox_ids

        return dlg_graph
