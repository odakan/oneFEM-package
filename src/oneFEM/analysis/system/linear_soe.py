##-----------------------------------------------------------------------##
#                                                                         #
#        #--oneFEM--#: One FEM software in a galaxy far far away          #
#                                                                         #
##-----------------------------------------------------------------------##
# LinearSOE / LinearSOESolver — OpenSees-pattern base classes.
#
# LinearSOE     — owns storage (SparseAssembler) + wires a solver
# LinearSOESolver — owns factorization logic, back-pointer to SOE
#
# Hierarchy:
#   System                — legacy solve-only base (FullGeneral, ProfileSPD)
#   LinearSOE(System)     — full interface: assembly + solve
#   LinearSOESolver       — pluggable solver backend
#     ├── SpSolve         — scipy.sparse.linalg.spsolve  (stateless)
#     └── UmfPackSolver   — scikits.umfpack  (in umfpack.py)

import numpy as np
from scipy.sparse.linalg import spsolve as _spsolve


class System(object):
    """Legacy base for solve-only systems (FullGeneral, ProfileSPD)."""
    def __init__(self, ID=-1):
        self._ID = ID

    def solve(self, A, b):
        raise NotImplementedError(
            "System.solve(): method must be implemented by subclasses.")


class LinearSOE(System):
    """Base class for systems that own both assembly and solving.

    Subclasses compose a SparseAssembler (storage) and a LinearSOESolver
    (factorization).  Analysis sees this unified interface:
      setSize, zeroA, addA, addM, addB, getK, getM, getB, getX, solve
    """
    def setSize(self, n, elem_dof_map, elements):
        raise NotImplementedError

    def zeroA(self):
        raise NotImplementedError

    def addA(self, ke, elem_id):
        raise NotImplementedError

    def addM(self, me, elem_id):
        raise NotImplementedError

    def addB(self, fe, dofs):
        raise NotImplementedError

    def getK(self):
        raise NotImplementedError

    def getM(self):
        raise NotImplementedError

    def getB(self):
        raise NotImplementedError

    def getX(self):
        raise NotImplementedError


class LinearSOESolver(object):
    """Base class for direct linear solvers wired to a LinearSOE.

    Interface:
      setLinearSOE(soe) — wire back-pointer to the owning SOE
      symbolic(K_csc)   — symbolic factorization (once per sparsity pattern)
      numeric(K_csc)    — numeric factorization (once per stiffness values)
      solve(K_csc, f)   — solve K x = f, returns x as ndarray
    """

    def setLinearSOE(self, soe):
        """Wire back-pointer to the owning LinearSOE."""
        self._soe = soe

    def symbolic(self, K_csc):
        """Symbolic factorization (sparsity-only, reuse across Newton iters)."""
        pass

    def numeric(self, K_csc):
        """Numeric factorization (values changed, pattern fixed)."""
        pass

    def solve(self, K_csc, f):
        """Solve K x = f.  Returns x as ndarray."""
        raise NotImplementedError


class SpSolve(LinearSOESolver):
    """scipy.sparse.linalg.spsolve — stateless, no symbolic/numeric split."""

    def solve(self, A, b):
        if hasattr(A, 'toarray'):
            return _spsolve(A.tocsc(), np.asarray(b))
        else:
            return np.linalg.solve(np.asarray(A), np.asarray(b))
