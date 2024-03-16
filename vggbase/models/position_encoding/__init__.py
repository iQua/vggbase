"""
The interface of position encoding methods. 
 It mainly includes:
    - absolute position encoding methods:
     -- BasicPositionEncoder
     -- SinusoidalPosition1DEncoder
     -- SinusoidalPosition2DEncoder
    
    - relative position encoding methods:
     -- RelativePosition1DEncoder
     -- RelativePosition2DEncoder
"""

from .absolute_position_encoding import BasicPositionEncoder, SinusoidalPosition1DEncoder, SinusoidalPosition2DEncoder
from .relative_position_encoding import RelativePosition1DEncoder, RelativePosition2DEncoder
