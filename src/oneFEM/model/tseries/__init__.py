# oneFEM/model/tseries/__init__.py

from .main import TSeries
from .constant_series import Constant
from .linear_series import Linear
from .path_series import Path

# delete modules imported from .py directories
del main
del constant_series
del linear_series
del path_series

__all__ = [
    "TSeries",
    "Constant",
    "Linear",
    "Path"
]