##-----------------------------------------------------------------------##
#                                                                         #
#        #--oneFEM--#: One FEM software in a galaxy far far away          #
#                                                                         #
##-----------------------------------------------------------------------##
# SparseAssembler — COO assembly with pre-computed sparsity pattern.
#   Pure data structure: no solve logic.  Composed by SparseGeneral / UmfPackSOE.

import numpy as np
from scipy.sparse import csr_matrix


class SparseAssembler(object):
    def __init__(self):
        self._n = 0
        self._rows = None       # int32 COO row indices
        self._cols = None       # int32 COO col indices
        self._data_K = None     # float64 COO values (K)
        self._data_M = None     # float64 COO values (M)
        self._B = None          # RHS vector
        self._elem_offsets = {}
        self._has_mass = False

    def setSize(self, n, elem_dof_map, elements):
        """Pre-compute COO sparsity pattern from element connectivity."""
        self._n = n
        self._B = np.zeros(n)
        self._has_mass = any(e.getMass() is not None for e in elements)

        rows_list = []
        cols_list = []
        offset = 0
        for elem in elements:
            dofs = elem_dof_map[elem._ID]
            n_local = len(dofs)
            n_entries = n_local * n_local
            ii = np.repeat(dofs, n_local)
            jj = np.tile(dofs, n_local)
            rows_list.append(ii)
            cols_list.append(jj)
            self._elem_offsets[elem._ID] = (offset, offset + n_entries)
            offset += n_entries

        self._rows = np.concatenate(rows_list).astype(np.int32)
        self._cols = np.concatenate(cols_list).astype(np.int32)
        self._data_K = np.zeros(offset, dtype=np.float64)
        if self._has_mass:
            self._data_M = np.zeros(offset, dtype=np.float64)

    def zeroA(self):
        """Zero matrix values, keeping sparsity pattern."""
        self._data_K[:] = 0.0
        if self._data_M is not None:
            self._data_M[:] = 0.0
        self._B[:] = 0.0

    def addA(self, ke, elem_id):
        """Scatter element stiffness into COO data."""
        start, end = self._elem_offsets[elem_id]
        self._data_K[start:end] += np.asarray(ke).ravel()

    def addM(self, me, elem_id):
        """Scatter element mass into COO data."""
        if self._data_M is None:
            return
        start, end = self._elem_offsets[elem_id]
        self._data_M[start:end] += np.asarray(me).ravel()

    def addB(self, fe, dofs):
        """Scatter force into RHS vector."""
        np.add.at(self._B, np.asarray(dofs, dtype=np.int32),
                  np.asarray(fe, dtype=np.float64))

    def getK(self):
        """Return assembled K as CSR matrix."""
        return csr_matrix((self._data_K, (self._rows, self._cols)),
                          shape=(self._n, self._n))

    def getM(self):
        """Return assembled M as CSR matrix (or None)."""
        if self._data_M is None:
            return None
        return csr_matrix((self._data_M, (self._rows, self._cols)),
                          shape=(self._n, self._n))

    def getB(self):
        """Return the RHS vector."""
        return self._B
