import numpy as np
from .main import Algorithm
from ..._systools.data import Vector

class Linear(Algorithm):
    def __init__(self, ID=-1):
        super().__init__(ID)

    def solve(self, model, uu, pp, integrator, system, test=None):
        """Single-pass solve using integrator interface.

        integrator.formTangent() -> integrator.formUnbalance() ->
        system.solve() -> integrator.update()
        """
        uu_idx = np.array(uu, dtype=int)
        pp_idx = np.array(pp, dtype=int)

        integrator.formTangent(model, 'current')
        integrator.formUnbalance(model)

        K_eff = integrator.getTangent()
        R = integrator.getUnbalance()

        K_uu = K_eff[np.ix_(uu_idx, uu_idx)]
        R_uu = R[uu_idx]

        dU_uu = system.solve(K_uu, R_uu)

        dU = np.zeros(len(R))
        dU[uu_idx] = dU_uu

        integrator.update(model, dU)

        # Reactions: F_int at fixed DOFs
        F_int = model.getInternalForce()
        F_data = np.asarray(model.F)
        F_data[pp_idx] = F_int[pp_idx]
        model.F = Vector(list(F_data))
