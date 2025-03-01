# oneFEM/model/material/uniaxial/__init__.py

# uniaxial (1D) materials
from .elastic import Elastic

# delete modules imported from .py directories
del elastic


__all__ = [
    "Elastic"
]