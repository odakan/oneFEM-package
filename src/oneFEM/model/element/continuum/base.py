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
from ...kinematics.continuum.cauchy.linear import LinearContinuumKinematics
from ...physics_family import PhysicsFamily
from ...._systools.data import Vector, Matrix
from ...._systools.backend import np


def compute_jacobian(dN_dxi, X_nodes):
    """Compute Jacobian matrix J and its determinant.

    J = dN_dxi^T @ X_nodes  (nDim x nDim)

    :param dN_dxi:   shape function natural derivatives (nNodes x nDim)
    :param X_nodes:  nodal coordinates (nNodes x nDim)
    :return: (J, detJ) — Jacobian matrix and its determinant
    """
    J = dN_dxi.T @ X_nodes
    detJ = np.linalg.det(J)
    return J, detJ


def compute_physical_derivatives(dN_dxi, X_nodes):
    """Compute shape function derivatives in physical coordinates.

    dN_dX = dN_dxi @ J^{-1}  (nNodes x nDim)

    :param dN_dxi:   shape function natural derivatives (nNodes x nDim)
    :param X_nodes:  nodal coordinates (nNodes x nDim)
    :return: (dN_dX, detJ) — physical derivatives and Jacobian determinant
    """
    J, detJ = compute_jacobian(dN_dxi, X_nodes)
    if abs(detJ) < 1e-30:
        raise ValueError("isoparametric: Jacobian determinant is zero or near-zero!")
    J_inv = np.linalg.inv(J)
    dN_dX = dN_dxi @ J_inv.T
    return dN_dX, detJ


class ContinuumElement(Element):
    """Base class for continuum (solid) elements.

    Option B architecture: single kinematics object, gp-indexed calls.

    Owns the integration loop. Subclasses provide:
      - _getGaussPoints()  -> list of (xi, eta, weight)
      - _getShapeDerivatives(xi, eta) -> dN_dxi (nNodes x nDim)
      - _nNodesPerElement, _nDimElement
    """

    physics_family = PhysicsFamily.CONTINUUM_CAUCHY

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
        self._committed_u_e = None
        self._committed_u_e_backup = None

    def _domain(self):
        """Initialize element geometry, allocate arrays, copy materials per GP."""
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
                                    gp_weights=list(zip(detJ_list, w_list)),
                                    element=self)

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

        # Track committed displacement for incremental formulations (UL)
        self._committed_u_e = Vector(shape=self._nDOF_total)
        self._committed_u_e_backup = Vector(shape=self._nDOF_total)

    def _update(self):
        """Extract nodal displacements, update kinematics and materials per GP."""
        nDim = self._nD
        nNodes = len(self._nodes)

        # Total displacement from original coordinates
        u_e = Vector(shape=nDim * nNodes)
        for i, nd in enumerate(self._nodes):
            u_nd = nd._getTrialDisp()
            for j in range(nDim):
                u_e[i * nDim + j] = u_nd[j]

        # For UL: kinematics expects incremental displacement from committed config
        if self._kinematics.needs_incremental_u:
            u_for_kin = Vector(init=(u_e.data - self._committed_u_e.data))
        else:
            u_for_kin = u_e

        # Pass 1: update kinematics at all GPs
        for gp in range(len(self._gp_data)):
            self._kinematics.update(gp, u_for_kin)
        self._kinematics.applyCorotFrame(u_for_kin)

        # Pass 2: push strain to materials via kinematics hook
        for gp in range(len(self._gp_data)):
            strain = self._kinematics.getStrain(gp)
            self._kinematics._setMaterialStrain(self._materials[gp], strain)

        self._buildStiffnessAndForce(u_for_kin)
        return 0

    def _buildStiffnessAndForce(self, u_e):
        """Assemble element K and f — delegated to kinematics integration loop."""
        self._k = self._kinematics.getK(self)
        self._f = self._kinematics.get_f_int(self)

    def _commit(self):
        """Commit kinematics and materials."""
        # Backup committed displacement before updating (for revert)
        self._committed_u_e_backup = Vector(init=self._committed_u_e.data.copy())

        X_current = self._buildCurrentCoords()
        self._kinematics.commitState(X_current=X_current)

        for gp in range(len(self._gp_data)):
            new_detJ = self._kinematics.getDetJ(gp)
            if new_detJ is not None:
                _, w = self._gp_data[gp]
                self._gp_data[gp] = (new_detJ, w)

        for mat in self._materials:
            mat._commitState()

        # Update committed displacement from committed node state
        nDim = self._nD
        for i, nd in enumerate(self._nodes):
            u_commit = nd._getCommitDisp()
            for j in range(nDim):
                self._committed_u_e[i * nDim + j] = u_commit[j]

        return 0

    def _revert(self):
        """Revert kinematics and materials."""
        self._kinematics.revertToLastCommit()
        for mat in self._materials:
            mat._revertToLastCommit()
        self._committed_u_e = Vector(init=self._committed_u_e_backup.data.copy())
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

    # Subclass hooks (legacy — subclasses override these)
    def _getGaussPoints(self):
        raise NotImplementedError("ContinuumElement._getGaussPoints()")

    def _getShapeDerivatives(self, xi, eta):
        raise NotImplementedError("ContinuumElement._getShapeDerivatives()")

    # ------------------------------------------------------------------
    # Element API (v2 architecture)
    # ------------------------------------------------------------------
    # Purely additive — does not change any existing data paths.
    # Kinematics continue to use the old initialize() path unchanged.
    # These methods provide a clean interface for future kinematics
    # refactoring (Phases 3+).

    # --- Geometry ---

    def get_N(self, xi):
        """Shape functions at natural coordinates xi. Delegates to subclass."""
        return self._get_shape_functions(xi)

    def get_dN_dxi(self, xi):
        """Shape function natural derivatives at xi. Delegates to subclass."""
        return self._get_shape_derivatives(xi)

    def get_jacobian(self, xi):
        """Jacobian matrix and determinant at natural coordinates xi.

        :param xi: tuple of natural coordinates
        :return: (J, detJ)
        """
        dN_dxi = self._get_shape_derivatives(xi)
        X_ref = self.get_coords_ref()
        return compute_jacobian(dN_dxi, X_ref)

    def get_dN_dX(self, xi):
        """Shape function derivatives in physical (reference) coordinates.

        :param xi: tuple of natural coordinates
        :return: numpy array (nNodes x nDim)
        """
        dN_dxi = self._get_shape_derivatives(xi)
        X_ref = self.get_coords_ref()
        dN_dX, _ = compute_physical_derivatives(dN_dxi, X_ref)
        return dN_dX

    def get_dN_dx(self, xi):
        """Shape function derivatives in current (deformed) coordinates.

        :param xi: tuple of natural coordinates
        :return: numpy array (nNodes x nDim)
        """
        dN_dxi = self._get_shape_derivatives(xi)
        x_cur = self.get_coords()
        dN_dx, _ = compute_physical_derivatives(dN_dxi, x_cur)
        return dN_dx

    def get_B(self, xi):
        """Standard linear B matrix at natural coordinates xi.

        :param xi: tuple of natural coordinates
        :return: Matrix (nVoigt x nDOF)
        """
        dN_dX = self.get_dN_dX(xi)
        return self._buildBMatrix(Matrix(init=dN_dX), self._nD)

    def get_B_NL(self, xi, u_e=None):
        """Nonlinear B matrix (consistent with Green-Lagrange strain).

        :param xi: tuple of natural coordinates
        :param u_e: element displacement vector (if None, uses current trial)
        :return: Matrix (nVoigt x nDOF)
        """
        if u_e is None:
            u_e = self.get_disp()
        dN_dX = self.get_dN_dX(xi)
        nDim = self._nD
        nNodes = len(self._nodes)
        u_data = u_e.reshape(nNodes, nDim) if hasattr(u_e, 'reshape') else np.asarray(u_e).reshape(nNodes, nDim)
        H = u_data.T @ dN_dX
        F = np.eye(nDim) + H
        nVoigt = 3 if nDim == 2 else 6
        nDOF = nDim * nNodes
        B = np.zeros((nVoigt, nDOF))

        if nDim == 2:
            for a in range(nNodes):
                for i in range(2):
                    col = a * 2 + i
                    B[0, col] = F[i, 0] * dN_dX[a, 0]
                    B[1, col] = F[i, 1] * dN_dX[a, 1]
                    B[2, col] = F[i, 0] * dN_dX[a, 1] + F[i, 1] * dN_dX[a, 0]
        else:
            for a in range(nNodes):
                for i in range(3):
                    col = a * 3 + i
                    B[0, col] = F[i, 0] * dN_dX[a, 0]
                    B[1, col] = F[i, 1] * dN_dX[a, 1]
                    B[2, col] = F[i, 2] * dN_dX[a, 2]
                    B[3, col] = F[i, 0] * dN_dX[a, 1] + F[i, 1] * dN_dX[a, 0]
                    B[4, col] = F[i, 1] * dN_dX[a, 2] + F[i, 2] * dN_dX[a, 1]
                    B[5, col] = F[i, 0] * dN_dX[a, 2] + F[i, 2] * dN_dX[a, 0]
        return Matrix(init=B)

    def get_F(self, xi, u_e=None):
        """Deformation gradient F = I + H at natural coordinates xi.

        :param xi: tuple of natural coordinates
        :param u_e: element displacement vector (if None, uses current trial)
        :return: numpy array (nDim x nDim)
        """
        if u_e is None:
            u_e = self.get_disp()
        dN_dX = self.get_dN_dX(xi)
        nDim = self._nD
        nNodes = len(self._nodes)
        u_data = u_e.reshape(nNodes, nDim) if hasattr(u_e, 'reshape') else np.asarray(u_e).reshape(nNodes, nDim)
        H = u_data.T @ dN_dX
        return np.eye(nDim) + H

    def get_H(self, xi, u_e=None):
        """Displacement gradient H at natural coordinates xi.

        :param xi: tuple of natural coordinates
        :param u_e: element displacement vector (if None, uses current trial)
        :return: numpy array (nDim x nDim)
        """
        if u_e is None:
            u_e = self.get_disp()
        dN_dX = self.get_dN_dX(xi)
        nDim = self._nD
        nNodes = len(self._nodes)
        u_data = u_e.reshape(nNodes, nDim) if hasattr(u_e, 'reshape') else np.asarray(u_e).reshape(nNodes, nDim)
        return u_data.T @ dN_dX

    def get_gauss_points(self):
        """Return list of gauss point coordinate tuples."""
        return self._getGaussPoints()

    # --- State ---

    def get_coords_ref(self):
        """Reference (undeformed) nodal coordinates as numpy array (nNodes x nDim)."""
        nNodes = len(self._nodes)
        nDim = self._nD
        X = np.zeros((nNodes, nDim))
        for i, nd in enumerate(self._nodes):
            coord = nd._getCoordinates()
            for j in range(nDim):
                X[i, j] = coord[j]
        return X

    def get_coords(self):
        """Current (deformed) nodal coordinates as numpy array (nNodes x nDim)."""
        nNodes = len(self._nodes)
        nDim = self._nD
        X = np.zeros((nNodes, nDim))
        for i, nd in enumerate(self._nodes):
            coord = nd._getCoordinates()
            u = nd._getTrialDisp()
            for j in range(nDim):
                X[i, j] = coord[j] + u[j]
        return X

    def get_disp(self):
        """Current trial nodal displacements as flat numpy array (nDOF_total,)."""
        nNodes = len(self._nodes)
        nDim = self._nD
        u = np.zeros(nDim * nNodes)
        for i, nd in enumerate(self._nodes):
            u_nd = nd._getTrialDisp()
            for j in range(nDim):
                u[i * nDim + j] = u_nd[j]
        return u

    def get_disp_committed(self):
        """Committed nodal displacements as flat numpy array (nDOF_total,)."""
        nNodes = len(self._nodes)
        nDim = self._nD
        u = np.zeros(nDim * nNodes)
        for i, nd in enumerate(self._nodes):
            u_nd = nd._getCommitDisp()
            for j in range(nDim):
                u[i * nDim + j] = u_nd[j]
        return u

    def get_stress(self, gp):
        """Return stress CTensor at gauss point gp."""
        return self._materials[gp].getStress()

    def get_tangent(self, gp):
        """Return material tangent CTensor at gauss point gp."""
        return self._materials[gp].getTangent()

    def get_material(self, gp):
        """Return material object at gauss point gp."""
        return self._materials[gp]

    def get_dof_indices(self):
        """Return list of global DOF indices for this element."""
        dofs = []
        for nd in self._nodes:
            dofs.extend(nd.getDOFs())
        return dofs

    # --- Dimensions ---

    def get_nDim(self):
        """Return spatial dimension (2 or 3)."""
        return self._nD

    def get_nNodes(self):
        """Return number of nodes."""
        return len(self._nodes)

    def get_nDOF_total(self):
        """Return total number of DOFs."""
        return self._nDOF_total

    def get_nVoigt(self):
        """Return Voigt vector size (3 for 2D, 6 for 3D)."""
        return 3 if self._nD == 2 else 6

    def get_nGP(self):
        """Return number of Gauss points."""
        return len(self._gp_data)

    # --- Integration data ---

    def get_thickness(self):
        """Return element thickness (1.0 for 3D)."""
        return self._thickness

    def get_gp_weight(self, gp):
        """Return (detJ, weight) tuple for Gauss point gp."""
        return self._gp_data[gp]

    # --- Enrichment stubs ---

    def get_strain_enrichment(self, xi):
        """Return enrichment contribution to Voigt strain vector.
        Default: None. Override in elements with incompatible modes."""
        return None

    def get_H_enrichment(self, xi):
        """Return incompatible mode enrichment to displacement gradient H.

        Default: no enrichment (returns None).
        Override in elements with incompatible modes (e.g., Hex8).
        """
        return None

    def get_G(self, xi):
        """Incompatible mode matrix. None for standard elements."""
        return None

    def get_nAlpha(self):
        """Number of incompatible mode parameters. 0 for standard elements."""
        return 0

    def get_B_bar(self, xi):
        """B-bar matrix. None unless B-bar is active."""
        return None

    # --- Subclass API hooks (v2) ---

    def _get_shape_functions(self, xi):
        """Return shape functions at natural coordinates xi.
        Subclasses must override. xi is a tuple of natural coordinates.
        """
        raise NotImplementedError("ContinuumElement._get_shape_functions()")

    def _get_shape_derivatives(self, xi):
        """Return shape function natural derivatives at xi.
        Subclasses must override. xi is a tuple of natural coordinates.
        """
        raise NotImplementedError("ContinuumElement._get_shape_derivatives()")

    @staticmethod
    def _buildBMatrix(dN_dX, nDim):
        """Build standard linear B matrix from shape function derivatives.

        :param dN_dX: Matrix (nNodes x nDim)
        :param nDim: spatial dimension (2 or 3)
        :return: Matrix (nVoigt x nDOF)
        """
        dN = dN_dX.data if hasattr(dN_dX, 'data') else np.asarray(dN_dX)
        nNodes = dN.shape[0]

        if nDim == 2:
            nVoigt = 3
            nDOF = 2 * nNodes
            B = np.zeros((nVoigt, nDOF))
            for i in range(nNodes):
                c = 2 * i
                dNi_dx = dN[i, 0]
                dNi_dy = dN[i, 1]
                B[0, c]     = dNi_dx
                B[1, c + 1] = dNi_dy
                B[2, c]     = dNi_dy
                B[2, c + 1] = dNi_dx
        else:
            nVoigt = 6
            nDOF = 3 * nNodes
            B = np.zeros((nVoigt, nDOF))
            for i in range(nNodes):
                c = 3 * i
                dNi_dx = dN[i, 0]
                dNi_dy = dN[i, 1]
                dNi_dz = dN[i, 2]
                B[0, c]     = dNi_dx
                B[1, c + 1] = dNi_dy
                B[2, c + 2] = dNi_dz
                B[3, c]     = dNi_dy
                B[3, c + 1] = dNi_dx
                B[4, c + 1] = dNi_dz
                B[4, c + 2] = dNi_dy
                B[5, c]     = dNi_dz
                B[5, c + 2] = dNi_dx

        return Matrix(init=B)

    def __repr__(self):
        return "ContinuumElement(ID={})".format(self._ID)
