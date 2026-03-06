# oneFEM/model/element/__init__.py

from .main import Element
from . import section
from . import beam
from . import shell
from . import zerolength
from . import truss
from . import kinematics
from . import continuum

# delete modules imported from .py directories
del main

__all__ = [
    "section",
    "Element",
    "beam",
    "shell",
    "zerolength",
    "truss",
    "kinematics",
    "continuum"
]
