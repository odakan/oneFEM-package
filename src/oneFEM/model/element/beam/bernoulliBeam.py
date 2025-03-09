##-----------------------------------------------------------------------##
#                                                                         #
#       #--oneFEM--#: One FEM software in a galaxy far far away           #
#                                                                         #
#                   Computational Mechanics 2022                          #
#                       University of Pavia                               #
#               Written by: Onur Deniz AKAN, IUSS Pavia                   #
#                         15 January 2022                                 #
#                                                                         #
##-----------------------------------------------------------------------##
#bernoulliBeam element sub-object definition
#   use bending and axial elastica to compute element stiffness matrix

import numpy as np
from ..main import Element


class BernoulliBeam(Element):
    """
    BernoulliBeam element class.
    Uses bending and axial elastica to compute element stiffness matrix.
    """

    def __init__(self, E, A, I, L):
        """
        Initialize BernoulliBeam properties.

        Parameters:
        E : float - Young's modulus
        A : float - Cross-sectional area
        I : float - Moment of inertia
        L : float - Element length
        """
        self.E = E
        self.A = A
        self.I = I
        self.L = L

    def compute_stiffness_matrix(self):
        """
        Compute the stiffness matrix for the beam element.

        Returns:
        K : numpy.ndarray - Element stiffness matrix
        """
        # Compute the local stiffness matrix for a Bernoulli beam element
        L = self.L
        E = self.E
        I = self.I
        A = self.A

        # Stiffness coefficients for axial and bending
        axial = E * A / L
        bending = E * I / (L**3)

        # Local stiffness matrix
        K = np.array([
            [ axial,     0,           0, -axial,     0,           0],
            [     0,  12*bending,  6*L*bending,      0, -12*bending,  6*L*bending],
            [     0,   6*L*bending,  4*L**2*bending, 0,  -6*L*bending,  2*L**2*bending],
            [-axial,     0,           0,  axial,     0,           0],
            [     0, -12*bending, -6*L*bending,      0,  12*bending, -6*L*bending],
            [     0,   6*L*bending,  2*L**2*bending, 0,  -6*L*bending,  4*L**2*bending]
        ])

        return K

    def add_element(self, data):
        """
        Add a BernoulliBeam element based on input data.

        Parameters:
        data : list - Element parameters
        """
        if len(data) != 4:
            raise ValueError("BernoulliBeam: Incorrect input length, expected 4 parameters.")

        self.E = data[0]
        self.A = data[1]
        self.I = data[2]
        self.L = data[3]
