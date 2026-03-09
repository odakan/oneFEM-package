##-----------------------------------------------------------------------##
#                                                                         #
#        #--oneFEM--#: One FEM software in a galaxy far far away          #
#                                                                         #
#                   Computational Mechanics 2022                          #
#                       University of Pavia                               #
#               Written by: Onur Deniz AKAN, IUSS Pavia                   #
#                                                                         #
##-----------------------------------------------------------------------##
# UmfPackSolver + UmfPackSOE
#
# UmfPackSolver(LinearSOESolver) — UMFPACK symbolic+numeric factorization
#   - Internal reordering disabled (UMFPACK_ORDERING_NONE) — RCM already did it
#   - Strategy: symmetric (UMFPACK_STRATEGY_SYMMETRIC)
#   - Verbosity: silent  (UMFPACK_PRL = 0)
#
# UmfPackSOE(LinearSOE) — composes SparseAssembler + UmfPackSolver
#   Pattern: setSize() → [zeroA() → addA() loop → solve()] per iteration

import numpy as np
from .linear_soe import LinearSOE, LinearSOESolver
from .sparse_assembler import SparseAssembler


class UmfPackSolver(LinearSOESolver):
    """UMFPACK direct solver with symbolic reuse."""

    def __init__(self):
        try:
            import scikits.umfpack as umf
            self._umf = umf
        except ImportError:
            raise ImportError(
                "UmfPackSolver requires scikits.umfpack. "
                "Install with: pip install scikit-umfpack")
        self._lu = None
        self._symbolic_done = False

    def symbolic(self, K_csc):
        """Create context, configure controls, run symbolic factorization."""
        umf = self._umf
        self._lu = umf.UmfpackContext()
        self._lu.control[umf.UMFPACK_ORDERING] = umf.UMFPACK_ORDERING_NONE
        self._lu.control[umf.UMFPACK_STRATEGY] = umf.UMFPACK_STRATEGY_SYMMETRIC
        self._lu.control[umf.UMFPACK_PRL] = 0
        self._lu.symbolic(K_csc)
        self._symbolic_done = True

    def numeric(self, K_csc):
        """Numeric factorization with current values."""
        self._lu.numeric(K_csc)

    def solve(self, A, b):
        """Solve A x = b.  Performs symbolic (once) and numeric per call."""
        if hasattr(A, 'tocsc'):
            A_csc = A.tocsc()
        else:
            from scipy.sparse import csc_matrix
            A_csc = csc_matrix(np.asarray(A))

        b_arr = np.asarray(b, dtype=np.float64)

        if not self._symbolic_done:
            self.symbolic(A_csc)

        self.numeric(A_csc)
        return self._lu.solve(
            self._umf.UMFPACK_A, A_csc, b_arr, autoTranspose=True)


class UmfPackSOE(LinearSOE):
    """LinearSOE that composes SparseAssembler + UmfPackSolver."""

    def __init__(self, sID=-1):
        super().__init__(sID)
        self._asm = SparseAssembler()
        self._solver = UmfPackSolver()
        self._solver.setLinearSOE(self)
        self._symbolic_done = False
        self._X = None

    def setSize(self, n, elem_dof_map, elements):
        self._asm.setSize(n, elem_dof_map, elements)
        self._X = np.zeros(n)
        self._symbolic_done = False

    def zeroA(self):
        self._asm.zeroA()
        if self._X is not None:
            self._X[:] = 0.0

    def addA(self, ke, elem_id):
        self._asm.addA(ke, elem_id)

    def addM(self, me, elem_id):
        self._asm.addM(me, elem_id)

    def addB(self, fe, dofs):
        self._asm.addB(fe, dofs)

    def getK(self):
        return self._asm.getK()

    def getM(self):
        return self._asm.getM()

    def getB(self):
        return self._asm.getB()

    def getX(self):
        return self._X

    def solve(self, A, b):
        """Solve A x = b via UmfPackSolver (symbolic reuse)."""
        x = self._solver.solve(A, b)
        n = self._asm._n
        if n > 0:
            self._X = np.zeros(n)
            self._X[:len(x)] = x
        return x
