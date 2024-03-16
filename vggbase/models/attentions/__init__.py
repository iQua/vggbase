"""
The interface of attentions for visual grounding.

One great article describes the attention:
https://vaclavkosar.com/ml/cross-attention-in-transformer-architecture


See a great tutorial shared by:
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html

 It mainly includes:
    -- SelfAttention
    -- UniDirectionalCrossAttention
    -- BiDirectionalCrossAttention
"""

from .self_attention import SelfAttention
from .unidirectional_cross_attention import UniDirectionalCrossAttention
from .bidirectional_cross_attention import BiDirectionalCrossAttention
from .ram_attention import ReconfiguredAttention
