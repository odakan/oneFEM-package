import numpy as np
from scipy.linalg import solve_banded
from .main import System

class ProfileSPD(System):
    def __init__(self, sID=-1):
        super().__init__(sID)

    def solve(self, A, b):
        A_np = np.asarray(A, dtype=float)
        b_np = np.asarray(b, dtype=float)
        n = A_np.shape[0]

        # Compute bandwidth (half-bandwidth)
        bw = 0
        for i in range(n):
            for j in range(i + 1, n):
                if A_np[i, j] != 0.0:
                    bw = max(bw, j - i)

        # Build banded storage: (2*bw+1, n) for symmetric banded
        ab = np.zeros((2 * bw + 1, n), dtype=float)
        for i in range(n):
            for j in range(max(0, i - bw), min(n, i + bw + 1)):
                ab[bw + i - j, j] = A_np[i, j]

        return solve_banded((bw, bw), ab, b_np)
