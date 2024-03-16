"""
The interface of regions operations for visual grounding.
 It mainly includes:
    - Patch operations
        -- PatchMerging
        -- PartialPatchMerging
        -- VisualTextPartialPatchMerging
        -- DynamicPatchEmbed
        -- PatchEmbed

    - Window operations
        -- WindowAttention
        -- window_count
        -- window_partition
        -- window_reverse
"""

from .patch_operations import PatchMerging, PartialPatchMerging, VisualTextPartialPatchMerging
from .patch_operations import AdaptivePatchMapper, PatchMapper

from .window_attention import WindowAttention

from .window_operations import window_count, window_partition, window_reverse
