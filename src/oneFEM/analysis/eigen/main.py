##-----------------------------------------------------------------------##
#                                                                         #
#        #--oneFEM--#: One FEM software in a galaxy far far away          #
#                                                                         #
#                   Computational Mechanics 2022                          #
#                       University of Pavia                               #
#               Written by: Onur Deniz AKAN, IUSS Pavia                   #
#                         15 January 2022                                 #
#                                                                         #
##-----------------------------------------------------------------------##
#
# Eigen solver for generalized eigenvalue problem: K*phi = lambda*M*phi
#

import numpy as np

class Eigen(object):
    def __init__(self, solver='genBandArpack'):
        self._solver = solver

    def solve(self, K, M, numModes):
        """Solve K*phi = lambda*M*phi for the smallest numModes eigenvalues.

        Parameters
        ----------
        K : numpy ndarray
            Stiffness matrix (nDOF x nDOF), symmetric positive semi-definite.
        M : numpy ndarray
            Mass matrix (nDOF x nDOF), symmetric positive definite.
        numModes : int
            Number of modes to compute.

        Returns
        -------
        eigenvalues : numpy array (numModes,)
            Eigenvalues (omega^2 values), sorted ascending.
        eigenvectors : numpy array (nDOF, numModes)
            Mass-normalized eigenvectors as columns.
        """
        K = np.asarray(K, dtype=float)
        M = np.asarray(M, dtype=float)
        n = K.shape[0]

        if numModes > n:
            raise ValueError(
                f"oneFEM.Eigen.solve() - requested {numModes} modes but only {n} DOFs available.")

        if self._solver == 'fullGenLapack':
            eigenvalues, eigenvectors = self._solve_full(K, M, numModes)
        elif self._solver == 'genBandArpack':
            try:
                eigenvalues, eigenvectors = self._solve_sparse(K, M, numModes)
            except Exception:
                eigenvalues, eigenvectors = self._solve_full(K, M, numModes)
        else:
            raise ValueError(
                f"oneFEM.Eigen.solve() - unknown solver '{self._solver}'. "
                "Use 'fullGenLapack' or 'genBandArpack'.")

        # Sort by ascending eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Mass-normalize: phi_i / sqrt(phi_i^T * M * phi_i)
        for i in range(numModes):
            phi = eigenvectors[:, i]
            mi = phi.dot(M.dot(phi))
            if mi > 0.0:
                eigenvectors[:, i] = phi / np.sqrt(mi)

        return eigenvalues, eigenvectors

    def _solve_full(self, K, M, numModes):
        """Dense symmetric generalized eigensolver (LAPACK)."""
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(K, M, subset_by_index=[0, numModes - 1])
        return eigenvalues, eigenvectors

    def _solve_sparse(self, K, M, numModes):
        """Sparse shift-invert eigensolver (ARPACK)."""
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csc_matrix
        K_sp = csc_matrix(K)
        M_sp = csc_matrix(M)
        eigenvalues, eigenvectors = eigsh(K_sp, k=numModes, M=M_sp,
                                          sigma=0.0, which='LM')
        return eigenvalues, eigenvectors
