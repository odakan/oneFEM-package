# oneFEM/analysis/numberer/__init__.py

from .main import Numberer
from .plain import Plain
from .reverse_cuthill_mckee import RevCuthillMcKee
from .rcm import RCM

# delete modules imported from .py directories
del main
del plain
del reverse_cuthill_mckee
del rcm

__all__ = [
    "Numberer",
    "Plain",
    "RevCuthillMcKee",
    "RCM"
]