##-----------------------------------------------------------------------##
#                                                                         #
#        #--oneFEM--#: One FEM software in a galaxy far far away          #
#                                                                         #
#                   Computational Mechanics 2022                          #
#                       University of Pavia                               #
#               Written by: Onur Deniz AKAN, IUSS Pavia                   #
#                                                                         #
##-----------------------------------------------------------------------##
# SparseGeneral — LinearSOE that composes SparseAssembler + SpSolve.
#   COO assembly with pre-computed sparsity pattern, CSR solve via
#   scipy.sparse.linalg.spsolve (SuperLU backend).
#
#   Pattern: setSize() → [zeroA() → addA() loop → solve()] per iteration

import numpy as np
from .linear_soe import LinearSOE, SpSolve
from .sparse_assembler import SparseAssembler


class SparseGeneral(LinearSOE):
    def __init__(self, sID=-1):
        super().__init__(sID)
        self._asm = SparseAssembler()
        self._solver = SpSolve()
        self._solver.setLinearSOE(self)
        self._X = None

    def setSize(self, n, elem_dof_map, elements):
        self._asm.setSize(n, elem_dof_map, elements)
        self._X = np.zeros(n)

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
        """Solve A x = b via SpSolve (scipy spsolve / SuperLU)."""
        x = self._solver.solve(A, b)
        n = self._asm._n
        if n > 0:
            self._X = np.zeros(n)
            self._X[:len(x)] = x
        return x
