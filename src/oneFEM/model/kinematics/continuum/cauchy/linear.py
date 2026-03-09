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
# Version: 0.3
#
# LinearContinuumKinematics (v2 architecture — element API path)
#   Small-strain, small-displacement formulation.
#   Strain = symmetric gradient of displacement (infinitesimal strain).
#
#   2D: eps = [eps_xx, eps_yy, gamma_xy]  (engineering shear)
#       Voigt COV: [eps_xx, eps_yy, eps_xy] where eps_xy = 0.5*gamma_xy
#
#   3D: eps = [eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_xz]
#       Voigt COV: all with factor 0.5 on shears

from .base import ContinuumKinematics
from ....._systools.data import Vector, Matrix
from ....._systools.data.ctensor import CTensor
from ....._systools.backend import np


class LinearContinuumKinematics(ContinuumKinematics):
    """Linear (infinitesimal strain) kinematics for continuum elements.

    v2 architecture: when initialized with element= kwarg, delegates B matrix
    computation to the element API. Stores zero element geometry.
    """

    formulation = 'linear'

    def __init__(self, bbar=False):
        self._nGP = None
        self._nDim = None
        self._nVoigt = None
        self._nDOF = None
        self._B = []
        self._strain = []
        self._K_geo_zero = None
        self._bbar = bbar
        self._B_bar = []

    def initialize(self, nGP, nDim, nNodes, dN_dX_list, **kwargs):
        """Initialize B matrices, either from element API or legacy path.

        If element= kwarg is provided, uses element API to build B matrices.
        Otherwise falls back to legacy path (dN_dX_list + _buildBMatrix).
        """
        element = kwargs.get('element', None)
        self._nGP = nGP
        self._nDim = nDim
        self._nVoigt = 3 if nDim == 2 else 6
        self._nDOF = nNodes * nDim

        self._B = []
        self._B_bar = []
        self._strain = [None] * nGP

        if element is not None:
            # v2 path: build B from element API
            gauss_points = element.get_gauss_points()
            for gp in range(nGP):
                gp_tuple = gauss_points[gp]
                xi = gp_tuple[:-1]  # strip weight
                B = element.get_B(xi)
                self._B.append(B)
        else:
            # Legacy path: build B from raw dN_dX
            for gp in range(nGP):
                B = self._buildBMatrix(dN_dX_list[gp], nDim)
                self._B.append(B)

        self._K_geo_zero = Matrix(shape=[self._nDOF, self._nDOF])

        # B-bar (Hughes 1980) for 3D volumetric locking
        gp_weights = kwargs.get('gp_weights', None)
        if self._bbar and nDim == 3:
            if gp_weights is None:
                raise ValueError("B-bar requires gp_weights kwarg from ContinuumElement._domain()")

            V_e = 0.0
            nVoigt = self._nVoigt
            nDOF = self._nDOF
            B_vol_bar_data = np.zeros((nVoigt, nDOF))
            for gp in range(nGP):
                B_std = self._B[gp].data
                B_vol_gp = self._extract_B_vol(B_std)
                detJ_gp, w_gp = gp_weights[gp]
                V_e += detJ_gp * w_gp
                B_vol_bar_data += B_vol_gp * detJ_gp * w_gp
            B_vol_bar_data /= V_e

            for gp in range(nGP):
                B_std = self._B[gp].data
                B_vol_gp = self._extract_B_vol(B_std)
                B_dev_gp = B_std - B_vol_gp
                self._B_bar.append(Matrix(init=B_dev_gp + B_vol_bar_data))

    def update(self, gp, u_e):
        """Compute eps = B @ u_e, cache as CTensor."""
        B = self.getBMatrix(gp)
        eps_vec = B @ u_e
        self._strain[gp] = CTensor(eps_vec.data.tolist(), self._nVoigt, CTensor.COV)

    def getStrain(self, gp):
        """Return cached strain CTensor (2nd order, COV) at GP."""
        return self._strain[gp]

    def getBMatrix(self, gp):
        """Return B matrix at GP. Returns B-bar when active (bbar=True, 3D)."""
        if self._bbar and self._B_bar:
            return self._B_bar[gp]
        return self._B[gp]

    def getF(self, gp):
        return None

    def getGeometricStiffness(self, gp, stress):
        """Linear formulation: return zero Matrix (not None)."""
        return self._K_geo_zero

    def getK(self, element):
        """Assemble element tangent stiffness K = sum_gp B^T C B dV.
        Linear: no geometric stiffness, no transform."""
        nDOF = element.get_nDOF_total()
        t = element._thickness
        K = Matrix(shape=[nDOF, nDOF])
        for gp in range(self._nGP):
            B = self.getBMatrix(gp)
            C_mat = element.get_tangent(gp).to_matrix()
            detJ, w = element._gp_data[gp]
            dV = detJ * w * t
            K += B.T @ C_mat @ B * dV
        return K

    def get_f_int(self, element):
        """Assemble element internal force f = sum_gp B^T sigma dV.
        Linear: no transform."""
        nDOF = element.get_nDOF_total()
        t = element._thickness
        f = Vector(shape=nDOF)
        for gp in range(self._nGP):
            B = self.getBMatrix(gp)
            sig_vec = element.get_stress(gp).to_vector()
            detJ, w = element._gp_data[gp]
            dV = detJ * w * t
            f += B.T @ sig_vec * dV
        return f

    def copy(self):
        """Return a fresh (uninitialized) copy."""
        return LinearContinuumKinematics(bbar=self._bbar)

    @staticmethod
    def _extract_B_vol(B_std):
        """Extract volumetric part of B matrix (3D only, Hughes 1980).

        B_vol rows 0-2 = mean of normal strain rows; rows 3-5 = 0 (shear).
        """
        B_vol = np.zeros_like(B_std)
        trace_row = (B_std[0] + B_std[1] + B_std[2]) / 3.0
        B_vol[0] = trace_row
        B_vol[1] = trace_row
        B_vol[2] = trace_row
        return B_vol

    @staticmethod
    def _buildBMatrix(dN_dX, nDim):
        """Build standard linear B matrix from shape function derivatives.

        :param dN_dX: Matrix (nNodes x nDim)
        :param nDim: spatial dimension (2 or 3)
        :return: Matrix (nVoigt x nDOF)

        Legacy method — kept for backward compatibility with copy()/test code.
        """
        dN = dN_dX.data
        nNodes = dN.shape[0]

        if nDim == 2:
            nVoigt = 3
            nDOF = 2 * nNodes
            B = np.zeros((nVoigt, nDOF))
            for i in range(nNodes):
                c = 2 * i
                dNi_dx = dN[i, 0]
                dNi_dy = dN[i, 1]
                B[0, c]     = dNi_dx        # eps_xx
                B[1, c + 1] = dNi_dy        # eps_yy
                B[2, c]     = dNi_dy         # gamma_xy
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
                B[0, c]     = dNi_dx          # eps_xx
                B[1, c + 1] = dNi_dy          # eps_yy
                B[2, c + 2] = dNi_dz          # eps_zz
                B[3, c]     = dNi_dy           # gamma_xy
                B[3, c + 1] = dNi_dx
                B[4, c + 1] = dNi_dz          # gamma_yz
                B[4, c + 2] = dNi_dy
                B[5, c]     = dNi_dz           # gamma_xz
                B[5, c + 2] = dNi_dx

        return Matrix(init=B)
