# oneFEM/model/element/__init__.py

from .main import Element
from . import section
from . import beam
from . import shell
from . import solid
from . import zerolength

# delete modules imported from .py directories
del main

__all__ = [
    "section",
    "Element",
    "beam",
    "shell",
    "solid",
    "zerolength"
]