# oneFEM/analysis/test/__init__.py

from .main import Test
from .unbalanced_load import NormUnbalance
from .displacement_increment import NormDispIncr
from .energy_increment import EnergyIncr


# delete modules imported from .py directories
del main
del unbalanced_load
del displacement_increment
del energy_increment

__all__ = [
    "Test",
    "NormUnbalance",
    "NormDispIncr",
    "EnergyIncr"
]