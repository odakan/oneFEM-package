import numpy as np
from .linear_soe import System

class FullGeneral(System):
    def __init__(self, sID=-1):
        super().__init__(sID)

    def solve(self, A, b):
        return np.linalg.solve(np.asarray(A), np.asarray(b))
