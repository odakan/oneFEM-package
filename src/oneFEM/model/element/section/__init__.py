# oneFEM/model/element/section/__init__.py

from .main import Section

# Section properties
from .rectangular import Rectangular
from .fiber_section import FiberSection
from .plate_fiber import PlateFiber

# delete modules imported from .py directories
del main
del rectangular
del fiber_section
del plate_fiber

__all__ = [
    "Section",
    "Rectangular",
    "FiberSection",
    "PlateFiber"
]