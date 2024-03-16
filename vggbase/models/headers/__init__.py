"""
The interface of headers for visual grounding.
 It mainly includes:
    -- DirectBoxRegGroundingHeader
    -- PseudoSegGroundingHeader
    -- DirectWindowRegGroundingHeader
"""

from .box_head import DirectBoxRegGroundingHeader
from .pseudo_segment_head import PseudoSegGroundingHeader
from .window_box_head import DirectWindowRegGroundingHeader
