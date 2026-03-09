# oneFEM/model/kinematics/shell/__init__.py

from .base import ShellKinematics

# delete modules imported from .py directories
del base

__all__ = ["ShellKinematics"]
