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
# Date: 09/03/2026
# Version: 0.2
#
# Hex8 — 8-node trilinear hexahedral element (3D)
#   Optional B-bar (Hughes 1980) for volumetric locking
#   Optional Wilson incompatible modes (Taylor et al. 1976) for shear locking
#     9 internal DOFs: 3 bubble functions x 3 displacement components
#     Center-point Jacobian correction for patch test compliance
#   Nodes: 8 x Node33 (3 translational DOFs each), total 24 DOFs
#   Integration: 2x2x2 Gauss (full)
#   Material: nDMaterial (type='3D')
#   Kinematics: injected, defaults to LinearContinuumKinematics

import warnings
from .base import ContinuumElement, compute_jacobian
from ...kinematics.continuum.cauchy.linear import LinearContinuumKinematics
from ...._systools.backend import np
from ...._systools.data import Vector, Matrix
from ...._systools.data.ctensor import CTensor


def _hex8_shape_functions(xi, eta, zeta):
    """8-node trilinear hexahedron shape functions.

    Node ordering (right-hand rule, CCW bottom then CCW top):
        7---6
       /|  /|
      4---5 |       z
      | 3-|-2       |  y
      |/  |/        | /
      0---1          x

    :param xi:   natural coordinate xi   in [-1, 1]
    :param eta:  natural coordinate eta  in [-1, 1]
    :param zeta: natural coordinate zeta in [-1, 1]
    :return: N array of shape (8,)
    """
    xi_n   = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=float)
    eta_n  = np.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=float)
    zeta_n = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=float)
    return 0.125 * (1.0 + xi_n * xi) * (1.0 + eta_n * eta) * (1.0 + zeta_n * zeta)


def _hex8_shape_derivatives(xi, eta, zeta):
    """Natural derivatives dN/d(xi, eta, zeta) for 8-node hexahedron.

    :param xi:   natural coordinate xi
    :param eta:  natural coordinate eta
    :param zeta: natural coordinate zeta
    :return: dN_dxi array of shape (8, 3) — rows are nodes, cols are [dN/dxi, dN/deta, dN/dzeta]
    """
    xi_n   = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=float)
    eta_n  = np.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=float)
    zeta_n = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=float)

    dN_dxi  = 0.125 * xi_n   * (1.0 + eta_n * eta) * (1.0 + zeta_n * zeta)
    dN_deta = 0.125 * eta_n  * (1.0 + xi_n * xi)   * (1.0 + zeta_n * zeta)
    dN_dzeta = 0.125 * zeta_n * (1.0 + xi_n * xi)   * (1.0 + eta_n * eta)

    return np.column_stack([dN_dxi, dN_deta, dN_dzeta])


def _hex8_gauss_points():
    """2x2x2 Gauss quadrature points and weights for hexahedron.

    :return: list of (xi, eta, zeta, weight) tuples, 8 points
    """
    g = 1.0 / np.sqrt(3.0)
    pts = []
    for zeta in [-g, g]:
        for eta in [-g, g]:
            for xi in [-g, g]:
                pts.append((xi, eta, zeta, 1.0))
    return pts


class Hex8(ContinuumElement):
    """8-node trilinear hexahedral element (3D).

    Enrichment options:
      - bbar=True: B-bar (Hughes 1980) for volumetric locking (nu -> 0.5)
      - incompatible=True: Wilson incompatible modes (Taylor et al. 1976)
        for shear locking in bending. 9 internal DOFs (3 bubble functions
        x 3 displacement components) with center-point Jacobian correction.

    Node ordering (right-hand rule, CCW bottom then CCW top):
        7---6
       /|  /|
      4---5 |       z
      | 3-|-2       |  y
      |/  |/        | /
      0---1          x

    :param tag: Element ID
    :param nodes: list of 8 Node33 instances
    :param material: nDMaterial template (type='3D')
    :param kinematics: ContinuumKinematics (default: LinearContinuumKinematics)
    :param rho: mass density (default: 0.0)
    :param body_force: [bx, by, bz] array (default: None)
    :param incompatible: enable Wilson incompatible modes (default: True)
    :param bbar: enable B-bar (default: not incompatible)
    """

    def __init__(self, tag, nodes, material, kinematics=None, rho=0.0,
                 body_force=None, incompatible=True, bbar=None):
        if kinematics is None:
            if bbar is None:
                bbar = not incompatible
            kinematics = LinearContinuumKinematics(bbar=bbar)
        elif bbar is None:
            bbar = False
        pf = self.physics_family
        if hasattr(kinematics, 'physics_family') and kinematics.physics_family != pf:
            warnings.warn(
                "Hex8({}): kinematics physics_family={} != element physics_family={}".format(
                    tag, kinematics.physics_family.value, pf.value), stacklevel=2)
        if hasattr(material, 'physics_family') and material.physics_family != pf:
            warnings.warn(
                "Hex8({}): material physics_family={} != element physics_family={}".format(
                    tag, material.physics_family.value, pf.value), stacklevel=2)
        super().__init__(tag, nodes, material, kinematics, thickness=1.0)
        self._rho = rho
        self._body_force = body_force

        self._incompatible = incompatible
        self._bbar = bbar

        # Wilson incompatible modes: 3 bubbles x 3 displacement components = 9 DOFs
        # alpha layout: [a1..a3] enrich u_x, [a4..a6] enrich u_y, [a7..a9] enrich u_z
        # Each group of 3 corresponds to modes M1(xi), M2(eta), M3(zeta)
        self._nAlpha = 9 if incompatible else 0
        self._alpha = np.zeros(self._nAlpha)
        self._alpha_commit = np.zeros(self._nAlpha)

        self._J0_inv = None
        self._G_list = None

    def _getGaussPoints(self):
        return _hex8_gauss_points()

    def _getShapeDerivatives(self, xi, eta, zeta):
        return _hex8_shape_derivatives(xi, eta, zeta)

    # ------------------------------------------------------------------
    # Incompatible mode precomputation
    # ------------------------------------------------------------------

    def _precompute_incompatible(self, X=None):
        """Compute center-point Jacobian inverse and G matrices at all GPs.

        :param X: (nNodes x nDim) nodal coordinates. If None, uses reference
                  coordinates (construction-time default in _domain()).

        Called once at _domain() with original coordinates. Never refreshed
        at commit — Wilson-Taylor enrichment uses original-config J₀ for
        all kinematics formulations (see _commit() docstring).
        """
        if X is None:
            X = self.get_coords_ref()
        dN_dxi_0 = _hex8_shape_derivatives(0.0, 0.0, 0.0)
        J0, _ = compute_jacobian(dN_dxi_0, X)
        self._J0_inv = np.linalg.inv(J0)
        gauss_pts = self._getGaussPoints()
        self._G_list = [self._build_G_matrix(*gp[:-1]) for gp in gauss_pts]

    def _get_dM_dX(self, xi, eta, zeta):
        """Physical gradients of 3 bubble modes using center-point Jacobian.

        Bubble modes: M_1 = 1-xi^2, M_2 = 1-eta^2, M_3 = 1-zeta^2

        :return: (3x3) array where row k = grad(M_k) = [dM_k/dx, dM_k/dy, dM_k/dz]
        """
        dM_dxi = np.diag([-2.0 * xi, -2.0 * eta, -2.0 * zeta])
        return dM_dxi @ self._J0_inv

    def _build_G_matrix(self, xi, eta, zeta):
        """Incompatible mode strain-displacement matrix G (6x9) in Voigt.

        9 modes: 3 bubbles x 3 displacement components.
        Columns 0-2: M1,M2,M3 enriching u_x
        Columns 3-5: M1,M2,M3 enriching u_y
        Columns 6-8: M1,M2,M3 enriching u_z

        Voigt ordering: [xx, yy, zz, xy, yz, xz] (matches _buildBMatrix).
        Uses center-point J0 (Taylor et al. 1976) for patch test compliance.
        """
        dM_dX = self._get_dM_dX(xi, eta, zeta)   # (3x3)
        G = np.zeros((6, 9))
        # Normal strains: eps_ii = du_i/dX_i
        for k in range(3):
            G[0, k]     = dM_dX[k, 0]   # eps_xx from u_x enrichment
            G[1, 3 + k] = dM_dX[k, 1]   # eps_yy from u_y enrichment
            G[2, 6 + k] = dM_dX[k, 2]   # eps_zz from u_z enrichment
        # Shear strains (Voigt rows 3=xy, 4=yz, 5=xz)
        for k in range(3):
            # gamma_xy = du_x/dy + du_y/dx
            G[3, k]     = dM_dX[k, 1]   # du_x/dy from u_x enrichment
            G[3, 3 + k] = dM_dX[k, 0]   # du_y/dx from u_y enrichment
            # gamma_yz = du_y/dz + du_z/dy
            G[4, 3 + k] = dM_dX[k, 2]   # du_y/dz from u_y enrichment
            G[4, 6 + k] = dM_dX[k, 1]   # du_z/dy from u_z enrichment
            # gamma_xz = du_x/dz + du_z/dx
            G[5, k]     = dM_dX[k, 2]   # du_x/dz from u_x enrichment
            G[5, 6 + k] = dM_dX[k, 0]   # du_z/dx from u_z enrichment
        return G

    # ------------------------------------------------------------------
    # Element API overrides for enrichment injection
    # ------------------------------------------------------------------

    def get_H(self, xi, u_e=None):
        """Displacement gradient H with incompatible mode enrichment.

        H_enriched = u^T @ dN_dX + alpha_mat @ dM_dX
        where alpha_mat is the 9-vector alpha reshaped to (3x3):
          [[a1,a2,a3], [a4,a5,a6], [a7,a8,a9]]
        """
        H = super().get_H(xi, u_e)
        if self._incompatible and self._J0_inv is not None:
            dM_dX = self._get_dM_dX(*xi)
            alpha_mat = self._alpha.reshape(3, 3)
            H = H + alpha_mat @ dM_dX
        return H

    def get_F(self, xi, u_e=None):
        """Deformation gradient F = I + H_enriched."""
        return np.eye(self._nD) + self.get_H(xi, u_e)

    def get_B_NL(self, xi, u_e=None):
        """Nonlinear B matrix using enriched F for incompatible modes."""
        if not self._incompatible or self._J0_inv is None:
            return super().get_B_NL(xi, u_e)
        F = self.get_F(xi, u_e)
        dN_dX = self.get_dN_dX(xi)
        nNodes = len(self._nodes)
        nDOF = 3 * nNodes
        B = np.zeros((6, nDOF))
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

    def get_H_enrichment(self, xi):
        """Return alpha_mat @ dM_dX — the incompatible mode part of H."""
        if not self._incompatible or self._J0_inv is None:
            return None
        dM_dX = self._get_dM_dX(*xi)
        alpha_mat = self._alpha.reshape(3, 3)
        return alpha_mat @ dM_dX

    def get_G(self, xi):
        """Return incompatible mode strain-displacement matrix G (6x9)."""
        if not self._incompatible:
            return None
        return self._build_G_matrix(*xi)

    def get_nAlpha(self):
        """Return number of incompatible mode parameters."""
        return self._nAlpha

    # ------------------------------------------------------------------
    # Element lifecycle
    # ------------------------------------------------------------------

    def _domain(self):
        super()._domain()
        if self._incompatible:
            self._precompute_incompatible()
        if self._rho > 0:
            self._buildMass()

    def _update(self):
        """Update element state with incompatible mode enrichment.

        J₀ and G_list are NOT refreshed here. They stay at the committed
        configuration throughout Newton iterations, consistent with Bathe's
        UL where all reference quantities refer to the last converged state.
        """
        if not self._incompatible:
            return super()._update()

        nDim = self._nD
        nNodes = len(self._nodes)

        u_e = Vector(shape=nDim * nNodes)
        for i, nd in enumerate(self._nodes):
            u_nd = nd._getTrialDisp()
            for j in range(nDim):
                u_e[i * nDim + j] = u_nd[j]

        if self._kinematics.needs_incremental_u:
            u_for_kin = Vector(init=(u_e.data - self._committed_u_e.data))
        else:
            u_for_kin = u_e

        for gp in range(len(self._gp_data)):
            self._kinematics.update(gp, u_for_kin)
        self._kinematics.applyCorotFrame(u_for_kin)

        formulation = getattr(self._kinematics, 'formulation', 'linear')
        for gp in range(len(self._gp_data)):
            if formulation == 'linear':
                B_np = np.asarray(self._kinematics.getBMatrix(gp))
                G = self._G_list[gp]
                eps = B_np @ np.asarray(u_e) + G @ self._alpha
                eps_ct = CTensor(eps.tolist(), 6, CTensor.COV)
                self._kinematics._setMaterialStrain(self._materials[gp], eps_ct)
            else:
                strain = self._kinematics.getStrain(gp)
                self._kinematics._setMaterialStrain(self._materials[gp], strain)

        self._buildStiffnessAndForce(u_for_kin)
        return 0

    def _buildStiffnessAndForce(self, u_e):
        """Assemble element K and f with incompatible mode static condensation.

        Wilson-Taylor static condensation (9 internal alpha DOFs):
          K* = K_uu - K_ua @ K_aa^-1 @ K_au
          f* = f_u  - K_ua @ K_aa^-1 @ f_a
        Alpha Newton correction: delta_alpha = -K_aa^-1 @ f_a
        """
        if not self._incompatible:
            super()._buildStiffnessAndForce(u_e)
            return

        nDOF = self._nDOF_total
        nAlpha = self._nAlpha
        nGP = len(self._gp_data)
        t = self._thickness

        K_uu_mat = np.zeros((nDOF, nDOF))
        K_ua = np.zeros((nDOF, nAlpha))
        K_aa = np.zeros((nAlpha, nAlpha))
        K_geo_total = np.zeros((nDOF, nDOF))
        f_u = np.zeros(nDOF)
        f_a = np.zeros(nAlpha)

        for gp in range(nGP):
            B = np.asarray(self._kinematics.getBMatrix(gp))
            G = self._G_list[gp]
            C = np.asarray(self._materials[gp].getTangent().to_matrix())
            sig = np.asarray(self._materials[gp].getStress().to_vector())
            detJ, w = self._gp_data[gp]
            dV = detJ * w * t

            K_uu_mat += (B.T @ C @ B) * dV
            K_ua += (B.T @ C @ G) * dV
            K_aa += (G.T @ C @ G) * dV
            f_u += (B.T @ sig) * dV
            f_a += (G.T @ sig) * dV

            stress_ct = self._materials[gp].getStress()
            Kg_mat = self._kinematics.getGeometricStiffness(gp, stress_ct)
            if Kg_mat is not None:
                K_geo_total += np.asarray(Kg_mat) * dV

        K_aa_inv = np.linalg.inv(K_aa)
        delta_alpha = -K_aa_inv @ f_a
        self._alpha = self._alpha + delta_alpha

        K_mat_condensed = K_uu_mat - K_ua @ K_aa_inv @ K_ua.T
        f_condensed = f_u - K_ua @ K_aa_inv @ f_a

        K_mat_M = Matrix(init=K_mat_condensed)
        f_V = Vector(init=f_condensed)
        K_global, f_global = self._kinematics.transformToGlobal(K_mat_M, f_V)

        self._k = Matrix(init=(np.asarray(K_global) + K_geo_total))
        self._f = Vector(init=np.asarray(f_global))

    def _commit(self):
        """Commit base state + incompatible mode amplitudes.

        J₀ is NEVER refreshed at commit — it stays at construction-time
        (original config) for all kinematics formulations. The Wilson-Taylor
        incompatible mode enrichment is an element-geometry artifact fix
        (shear locking), not a deformation-dependent quantity. Refreshing
        J₀ to committed config introduces configuration-dependent errors
        in alpha that accumulate non-monotonically with load steps.

        Validated: with original-config J₀ for all formulations, TL==UL
        to <0.01% at all mesh sizes with monotonic convergence under
        refinement. With committed-config J₀ for UL, gap was 11-20%
        and non-monotonic.
        """
        result = super()._commit()
        if self._incompatible:
            self._alpha_commit = self._alpha.copy()
        return result

    def _revert(self):
        """Revert base state + incompatible mode amplitudes."""
        result = super()._revert()
        if self._incompatible:
            self._alpha = self._alpha_commit.copy()
        return result

    # ------------------------------------------------------------------
    # Mass matrix
    # ------------------------------------------------------------------

    def _buildMass(self):
        """Consistent mass matrix: M = sum_gp rho * N_mat^T @ N_mat * detJ * w"""
        nDOF = self._nDOF_total
        M = np.zeros((nDOF, nDOF))
        gauss_pts = self._getGaussPoints()

        for gp_idx, gp_tuple in enumerate(gauss_pts):
            *coords, w = gp_tuple
            N = _hex8_shape_functions(*coords)
            detJ, _ = self._gp_data[gp_idx]

            N_mat = np.zeros((3, nDOF))
            for a in range(8):
                N_mat[0, 3*a]     = N[a]
                N_mat[1, 3*a + 1] = N[a]
                N_mat[2, 3*a + 2] = N[a]

            M += self._rho * (N_mat.T @ N_mat) * detJ * w

        self._m = Matrix(init=M)

    # ------------------------------------------------------------------
    # v2 element API hooks
    # ------------------------------------------------------------------

    def _get_shape_functions(self, xi):
        return _hex8_shape_functions(*xi)

    def _get_shape_derivatives(self, xi):
        return _hex8_shape_derivatives(*xi)

    def __repr__(self):
        flags = []
        if self._incompatible:
            flags.append('incompatible')
        if self._bbar:
            flags.append('bbar')
        flag_str = ', '.join(flags) if flags else 'standard'
        return "Hex8(ID={}, {})".format(self._ID, flag_str)
