import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from .main import System

class UmfPack(System):
    def __init__(self, sID=-1):
        super().__init__(sID)
        self._use_umfpack = False
        try:
            from scipy.sparse.linalg import use_solver
            import scikits.umfpack
            use_solver(useUmfpack=True)
            self._use_umfpack = True
        except (ImportError, Exception):
            pass

    def solve(self, A, b):
        A_np = np.asarray(A, dtype=float)
        b_np = np.asarray(b, dtype=float)
        A_sparse = csc_matrix(A_np)
        return spsolve(A_sparse, b_np, use_umfpack=self._use_umfpack)
