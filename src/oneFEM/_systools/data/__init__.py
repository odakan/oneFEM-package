# oneFEM/_systools/data/__init__.py

# Import submodules to make them accessible at the top level if desired
from .vector import Vector   # Import vector module
from .matrix import Matrix   # Import matrix module
from .tensor import Tensor   # Import tensor module
from .ctensor import CTensor # Import ctensor module

# delete modules imported from .py directories
del vector
del matrix
del tensor
del ctensor

__all__ = ["Vector", "Matrix", "Tensor", "CTensor"]