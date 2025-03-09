# oneFEM/model/pattern/__init__.py

from .main import Pattern
from .plain import Plain
from .uniform_excitation import UniformExcitation

# delete modules imported from .py directories
del main
del plain
del uniform_excitation

__all__ = [
    "Pattern",
    "Plain",
    "UniformExcitation"
]