# oneFEM/model/material/nD/__init__.py

# nD materials
from .elastic_isotropic import ElasticIsotropic

# delete modules imported from .py directories
del elastic_isotropic

__all__ = [
    "ElasticIsotropic"
]