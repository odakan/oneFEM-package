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
# Author: Onur Deniz Akan
# Date: 05/03/2026
# Version: 0.2
#
# ContinuumElement base class (Option B: single kinematics, gp-indexed)
#   Base for isoparametric continuum (solid) elements.
#   Owns the integration loop. Subclasses define topology (shape functions, GP layout).

from ..main import Element
from ..kinematics.continuum.linear import LinearContinuumKinematics
from ...._systools.data import Vector, Matrix
from ...._systools.backend import np


class ContinuumElement(Element):
    """Base class for continuum (solid) elements.

    Option B architecture: single kinematics object, gp-indexed calls.

    Owns the integration loop. Subclasses provide:
      - _getGaussPoints()  -> list of (xi, eta, weight)
      - _getShapeDerivatives(xi, eta) -> dN_dxi (nNodes x nDim)
      - _nNodesPerElement, _nDimElement
    """

    def __init__(self, tag, nodes, material, kinematics=None, thickness=1.0):
        super().__init__(tag)
        self._nodes = list(nodes)
        self._mat_template = material
        self._thickness = float(thickness)
        self._kin_template = kinematics if kinematics is not None else LinearContinuumKinematics()

        # Set during _domain()
        self._materials = []
        self._kinematics = None      # single kinematics object (Option B)
        self._gp_data = []           # list of (detJ, weight) per GP
        self._nDOF_total = 0
        self._m = None               # mass matrix, built by subclass if needed

    def _domain(self):
        """Initialize element geometry, allocate arrays, copy materials per GP."""
        from .isoparametric import compute_physical_derivatives

        nNodes = len(self._nodes)
        nDim = self._nodes[0].getND()
        self._nD = nDim

        nDOF_per_node = self._nodes[0].getNDOF()
        self._nDOF = Vector([nDOF_per_node] * nNodes, dtype=int)
        self._nDOF_total = nDOF_per_node * nNodes

        # Extract nodal coordinates (nNodes x nDim)
        X_nodes = np.zeros((nNodes, nDim))
        for i, nd in enumerate(self._nodes):
            coord = nd._getCoordinates()
            for j in range(nDim):
                X_nodes[i, j] = coord[j]

        gauss_points = self._getGaussPoints()
        nGP = len(gauss_points)

        # Compute dN_dX at each GP -> list of Matrix
        dN_dX_list = []
        dN_dxi_list = []
        detJ_list = []
        w_list = []
        for gp_tuple in gauss_points:
            *coords, w = gp_tuple
            dN_dxi = self._getShapeDerivatives(*coords)
            dN_dxi_list.append(dN_dxi)
            dN_dX, detJ = compute_physical_derivatives(dN_dxi, X_nodes)
            dN_dX_list.append(Matrix(init=dN_dX))
            detJ_list.append(detJ)
            w_list.append(w)

        self._gp_data = list(zip(detJ_list, w_list))

        # Single kinematics — initialize with all GP data
        self._kinematics = self._kin_template.copy()
        self._kinematics.initialize(nGP, nDim, nNodes, dN_dX_list,
                                    dN_dxi_list=dN_dxi_list,
                                    X_ref=Matrix(init=X_nodes),
                                    gp_weights=list(zip(detJ_list, w_list)))

        # Per-GP materials
        self._materials = [self._mat_template.getCopy() for _ in range(nGP)]

        # Initial stiffness
        u_zero = Vector(shape=self._nDOF_total)
        nDOF = self._nDOF_total
        t = self._thickness
        K0 = Matrix(shape=[nDOF, nDOF])
        for gp in range(nGP):
            self._kinematics.update(gp, u_zero)
        self._kinematics.applyCorotFrame(u_zero)
        for gp in range(nGP):
            B = self._kinematics.getBMatrix(gp)
            C_mat = self._materials[gp].getInitialTangent().to_matrix()
            detJ, w = self._gp_data[gp]
            dV = detJ * w * t
            K0 += B.T @ C_mat @ B * dV
        self._k = K0
        self._f = Vector(shape=nDOF)

    def _update(self):
        """Extract nodal displacements, update kinematics and materials per GP."""
        nDim = self._nD
        nNodes = len(self._nodes)

        # NOTE: u_e is the TOTAL displacement from the original coordinates.
        # For TL kinematics this is correct (reference = original config).
        # For UL kinematics this is wrong: UL updates dN_dX to the last
        # committed config, so it expects incremental displacement, not total.
        # The error is invisible for rectangular elements (dN_dX_updated ~=
        # dN_dX_original) but produces wrong strains for non-rectangular
        # elements (arch, annular meshes). See docs/known_issues.md.
        u_e = Vector(shape=nDim * nNodes)
        for i, nd in enumerate(self._nodes):
            u_nd = nd._getTrialDisp()
            for j in range(nDim):
                u_e[i * nDim + j] = u_nd[j]

        # Pass 1: update kinematics at all GPs
        for gp in range(len(self._gp_data)):
            self._kinematics.update(gp, u_e)
        self._kinematics.applyCorotFrame(u_e)

        # Pass 2: push strain to materials
        for gp in range(len(self._gp_data)):
            strain = self._kinematics.getStrain(gp)
            self._materials[gp]._setTrialStrain(strain)

        self._buildStiffnessAndForce(u_e)
        return 0

    def _buildStiffnessAndForce(self, u_e):
        """Assemble element K and f from all GPs."""
        nDOF = self._nDOF_total
        t = self._thickness

        K_mat = Matrix(shape=[nDOF, nDOF])
        K_geo = Matrix(shape=[nDOF, nDOF])
        f = Vector(shape=nDOF)

        for gp in range(len(self._gp_data)):
            B = self._kinematics.getBMatrix(gp)
            C_mat = self._materials[gp].getTangent().to_matrix()
            sig_vec = self._materials[gp].getStress().to_vector()
            detJ, w = self._gp_data[gp]
            dV = detJ * w * t

            K_mat += B.T @ C_mat @ B * dV
            f += B.T @ sig_vec * dV

            Kg = self._kinematics.getGeometricStiffness(gp, self._materials[gp].getStress())
            K_geo += Kg * dV

        K_mat, f = self._kinematics.transformToGlobal(K_mat, f)
        self._k = K_mat + K_geo
        self._f = f

    def _commit(self):
        """Commit kinematics and materials."""
        X_current = self._buildCurrentCoords()
        self._kinematics.commitState(X_current=X_current)

        for gp in range(len(self._gp_data)):
            new_detJ = self._kinematics.getDetJ(gp)
            if new_detJ is not None:
                _, w = self._gp_data[gp]
                self._gp_data[gp] = (new_detJ, w)

        for mat in self._materials:
            mat._commitState()
        return 0

    def _revert(self):
        """Revert kinematics and materials."""
        self._kinematics.revertToLastCommit()
        for mat in self._materials:
            mat._revertToLastCommit()
        return 0

    def _buildCurrentCoords(self):
        """Build Matrix of current nodal positions: X_ref + u_committed."""
        nNodes = len(self._nodes)
        nDim = self._nD
        X = np.zeros((nNodes, nDim))
        for i, nd in enumerate(self._nodes):
            coord = nd._getCoordinates()
            u_commit = nd._getCommitDisp()
            for j in range(nDim):
                X[i, j] = coord[j] + u_commit[j]
        return Matrix(init=X)

    def getMass(self):
        if self._m is None:
            return Matrix(shape=[self._nDOF_total, self._nDOF_total])
        return self._m

    def getDamp(self):
        return None

    def getInitialStiff(self):
        """Stiffness from initial tangent."""
        nDOF = self._nDOF_total
        t = self._thickness

        K0 = Matrix(shape=[nDOF, nDOF])
        u_zero = Vector(shape=self._nDOF_total)

        for gp in range(len(self._gp_data)):
            self._kinematics.update(gp, u_zero)
        self._kinematics.applyCorotFrame(u_zero)
        for gp in range(len(self._gp_data)):
            B = self._kinematics.getBMatrix(gp)
            C_mat = self._materials[gp].getInitialTangent().to_matrix()
            detJ, w = self._gp_data[gp]
            dV = detJ * w * t
            K0 += B.T @ C_mat @ B * dV

        return K0

    # Subclass hooks
    def _getGaussPoints(self):
        raise NotImplementedError("ContinuumElement._getGaussPoints()")

    def _getShapeDerivatives(self, xi, eta):
        raise NotImplementedError("ContinuumElement._getShapeDerivatives()")

    def __repr__(self):
        return "ContinuumElement(ID={})".format(self._ID)
