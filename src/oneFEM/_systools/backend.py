"""
backend.py - Array backend for oneFEM.

Single import point for the array library used by all data wrappers
(Vector, Matrix, CTensor) and numerical routines.

Currently: numpy (CPU).
Future: swap this one import to retarget hardware (CuPy, JAX, MLX, etc.).

Usage throughout oneFEM:
    from oneFEM._systools.backend import np
"""

import numpy as np

__all__ = ["np"]
