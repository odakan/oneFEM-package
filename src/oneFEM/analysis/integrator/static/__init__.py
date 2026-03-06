# oneFEM/analysis/integrator/static/__init__.py

from .load_control import LoadControl
from .displacement_control import DispControl

# delete modules imported from .py directories
del load_control
del displacement_control


__all__ = [
    "LoadControl",
    "DispControl",
]
