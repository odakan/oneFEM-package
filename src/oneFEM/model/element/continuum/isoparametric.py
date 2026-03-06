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
# Version: 0.1
#
# Isoparametric utility functions for continuum elements.
# Pure functions (no state). All return numpy arrays.

import numpy as np


def quad4_shape_functions(xi, eta):
    """4-node bilinear quad shape functions.

    Node ordering (CCW):
        3 --- 2
        |     |
        0 --- 1

    Natural coordinates: xi, eta in [-1, 1].

    :param xi:  natural coordinate xi
    :param eta: natural coordinate eta
    :return: N array of shape (4,)
    """
    return 0.25 * np.array([
        (1.0 - xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 + eta),
        (1.0 - xi) * (1.0 + eta),
    ])


def quad4_shape_derivatives(xi, eta):
    """Natural derivatives dN/d(xi, eta) for 4-node bilinear quad.

    :param xi:  natural coordinate xi
    :param eta: natural coordinate eta
    :return: dN_dxi array of shape (4, 2) — rows are nodes, cols are [dN/dxi, dN/deta]
    """
    return 0.25 * np.array([
        [-(1.0 - eta), -(1.0 - xi)],
        [ (1.0 - eta), -(1.0 + xi)],
        [ (1.0 + eta),  (1.0 + xi)],
        [-(1.0 + eta),  (1.0 - xi)],
    ])


def quad4_gauss_points():
    """2x2 Gauss quadrature points and weights for bilinear quad.

    :return: list of (xi, eta, weight) tuples, 4 points
    """
    g = 1.0 / np.sqrt(3.0)
    return [
        (-g, -g, 1.0),
        ( g, -g, 1.0),
        ( g,  g, 1.0),
        (-g,  g, 1.0),
    ]


def hex8_shape_functions(xi, eta, zeta):
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


def hex8_shape_derivatives(xi, eta, zeta):
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


def hex8_gauss_points():
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
