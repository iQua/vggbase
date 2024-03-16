"""
This package contains multiple well-known and classical masking
machanisms, including:

1. Square masking
2. Grid masking
3. Block masking
4. Random masking

For the comparsion of different masking mechanisms, please access the paper titled
"SimMIM: a Simple Framework for Masked Image Modeling".

"""

from mask_mechanism import square_masking
from mask_mechanism import grid_masking
from mask_mechanism import blockwise_masking
from mask_mechanism import blockwise_gmm_masking
from mask_mechanism import random_masking
