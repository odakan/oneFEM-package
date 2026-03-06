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
# ElasticIsotropic nDMaterial
#   Linear elastic isotropic material for 2D and 3D continuum elements.
#   Supports PlaneStress, PlaneStrain, and 3D formulations.
#
#   Stress-strain: sigma = C : epsilon (CTensor double-dot product)
#
#   PlaneStress (3x3):
#     C = E/(1-nu^2) * [[1,  nu, 0           ],
#                        [nu, 1,  0           ],
#                        [0,  0,  (1-nu)/2    ]]
#
#   PlaneStrain (3x3):
#     C = E/((1+nu)(1-2nu)) * [[1-nu,  nu,   0           ],
#                               [nu,    1-nu, 0           ],
#                               [0,     0,    (1-2nu)/2   ]]
#
#   3D (6x6):
#     C = E/((1+nu)(1-2nu)) * [[1-nu, nu,   nu,   0,          0,          0         ],
#                               [nu,   1-nu, nu,   0,          0,          0         ],
#                               [nu,   nu,   1-nu, 0,          0,          0         ],
#                               [0,    0,    0,    (1-2nu)/2,  0,          0         ],
#                               [0,    0,    0,    0,          (1-2nu)/2,  0         ],
#                               [0,    0,    0,    0,          0,          (1-2nu)/2 ]]

from .main import nDMaterial
from ...._systools.data.ctensor import CTensor


class ElasticIsotropic(nDMaterial):
    """Linear elastic isotropic nDMaterial.

    :param mat_id: Material ID
    :param E: Young's modulus
    :param nu: Poisson's ratio
    :param type: 'PlaneStress', 'PlaneStrain', or '3D'
    """

    def __init__(self, mat_id, E, nu, type='PlaneStress'):
        super().__init__(mat_id)
        self._E = float(E)
        self._nu = float(nu)
        self._type = type

        # Build initial tangent (constant for linear elastic)
        self._C0 = self._buildTangent()
        self._C = self._C0.copy()

        # Voigt size
        self._nVoigt = 3 if type != '3D' else 6

        # Trial and committed strain/stress
        self._eps = CTensor(self._nVoigt, CTensor.COV)
        self._sig = CTensor(self._nVoigt, CTensor.CONTR)
        self._eps_commit = CTensor(self._nVoigt, CTensor.COV)
        self._sig_commit = CTensor(self._nVoigt, CTensor.CONTR)

    def _buildTangent(self):
        """Build the elastic stiffness CTensor (4th order, CONTR)."""
        E = self._E
        nu = self._nu

        if self._type == 'PlaneStress':
            # 3x3 CONTR
            factor = E / (1.0 - nu * nu)
            data = [factor,        factor * nu,   0.0,
                    factor * nu,   factor,        0.0,
                    0.0,           0.0,           factor * (1.0 - nu) / 2.0]
            return CTensor(data, 3, 3, CTensor.CONTR)

        elif self._type == 'PlaneStrain':
            # 3x3 CONTR
            factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
            data = [factor * (1.0 - nu),  factor * nu,          0.0,
                    factor * nu,          factor * (1.0 - nu),  0.0,
                    0.0,                  0.0,                  factor * (1.0 - 2.0 * nu) / 2.0]
            return CTensor(data, 3, 3, CTensor.CONTR)

        elif self._type == '3D':
            # 6x6 CONTR
            factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
            G = factor * (1.0 - 2.0 * nu) / 2.0
            a = factor * (1.0 - nu)
            b = factor * nu
            data = [a, b, b, 0, 0, 0,
                    b, a, b, 0, 0, 0,
                    b, b, a, 0, 0, 0,
                    0, 0, 0, G, 0, 0,
                    0, 0, 0, 0, G, 0,
                    0, 0, 0, 0, 0, G]
            return CTensor(data, 6, 6, CTensor.CONTR)
        else:
            raise ValueError("ElasticIsotropic: unknown type '{}'. Use 'PlaneStress', 'PlaneStrain', or '3D'.".format(self._type))

    def __repr__(self):
        return "ElasticIsotropic(ID={}, E={}, nu={}, type='{}')".format(
            self._Material__ID, self._E, self._nu, self._type)

    def _getClassType(self):
        return "ElasticIsotropic nDMaterial"

    def _setTrialStrain(self, strain):
        """Accept CTensor (2nd order, COV). Compute stress = C : strain."""
        self._eps = strain.copy() if isinstance(strain, CTensor) else strain
        # sigma = C ^ epsilon (4th-order ^ 2nd-order -> 2nd-order)
        self._sig = self._C.__xor__(self._eps)

    def getStrain(self):
        return self._eps.copy()

    def getStress(self):
        return self._sig.copy()

    def getTangent(self):
        return self._C.copy()

    def getInitialTangent(self):
        return self._C0.copy()

    def _commitState(self):
        self._eps_commit = self._eps.copy()
        self._sig_commit = self._sig.copy()
        return 0

    def _revertToLastCommit(self):
        self._eps = self._eps_commit.copy()
        self._sig = self._sig_commit.copy()
        return 0

    def _revertToStart(self):
        self._eps = CTensor(self._nVoigt, CTensor.COV)
        self._sig = CTensor(self._nVoigt, CTensor.CONTR)
        self._eps_commit = CTensor(self._nVoigt, CTensor.COV)
        self._sig_commit = CTensor(self._nVoigt, CTensor.CONTR)
        self._C = self._C0.copy()
        return 0

    def getCopy(self):
        return ElasticIsotropic(self._Material__ID, self._E, self._nu, self._type)
