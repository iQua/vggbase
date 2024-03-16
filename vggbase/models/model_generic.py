"""
Generic components for visual grounding models.
"""

from typing import Optional
from dataclasses import dataclass

from vggbase.boxes.bbox_generic import BaseVGModelBBoxes
from vggbase.datasets.data_generic import BaseVGCollatedSamples
from vggbase.utils.generic_components import FieldFrozenContainer


@dataclass
class BaseVGModelInput(BaseVGCollatedSamples):
    """
    Base class for inputs of visual grounding models.

    Args:
        additional_input: A `FieldFrozenContainer` for including the
         possible additional input.
    """

    additional_input: Optional[FieldFrozenContainer] = None


@dataclass
class BaseVGModelOutput(BaseVGModelBBoxes):
    """
    Base class for outputs of visual grounding models.

    Args:
        additional_output: A `FieldFrozenContainer` for including the
         possible additional output.
    """

    additional_output: Optional[FieldFrozenContainer] = None
