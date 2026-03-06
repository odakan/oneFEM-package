# oneFEM/model/element/zerolength/__init__.py

# Zero-length elements
from .zerolength import ZeroLength
from .zerolength_contact import ZLContact

# delete modules imported from .py directories
del zerolength_contact
del zerolength


__all__ = [
    "ZeroLength",
    "ZLContact",
]