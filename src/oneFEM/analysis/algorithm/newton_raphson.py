import numpy as np
from .main import Algorithm
from ..._systools.data import Vector

class Newton(Algorithm):
    def __init__(self, ID=-1, tangent='current', line_search=False):
        super().__init__(ID)
        self._tangent = tangent  # 'current' or 'initial'
        self._line_search = line_search

    def solve(self, model, uu, pp, integrator, system, test=None):
        """Newton-Raphson iteration loop using integrator interface.

        Loop: formUnbalance -> test -> formTangent -> solve -> update -> repeat
        Works for static (LoadControl) and dynamic (Newmark) identically.

        Line search (optional): when the full Newton correction doesn't reduce
        the residual norm, bisect the step size until improvement is found.
        This prevents oscillation with EPP materials where zero tangent causes
        the correction to overshoot across the yield surface.
        """
        if test is None:
            raise ValueError("Newton.solve(): convergence test is required.")

        uu_idx = np.array(uu, dtype=int)
        pp_idx = np.array(pp, dtype=int)

        # Initial unbalance
        integrator.formUnbalance(model)

        test.start()

        for iteration in range(test.maxIter):
            # Test convergence on current residual
            R = integrator.getUnbalance()
            R_uu = R[uu_idx]
            norm_R = np.linalg.norm(R_uu)
            result = test.test(norm_R)

            if result == 0:
                # Converged — set reactions
                F_int = model.getInternalForce()
                F_data = np.asarray(model.F)
                F_data[pp_idx] = F_int[pp_idx]
                model.F = Vector(list(F_data))
                return

            if result == -2:
                raise RuntimeError(
                    "Newton.solve(): failed to converge in {} iterations, "
                    "norm = {:.6e}".format(test.maxIter, norm_R))

            # Form tangent
            integrator.formTangent(model, self._tangent)

            # Extract free-DOF partition
            K_eff = integrator.getTangent()
            K_uu = K_eff[np.ix_(uu_idx, uu_idx)]

            # Re-read residual (formTangent may have reassembled)
            R = integrator.getUnbalance()
            R_uu = R[uu_idx]

            # Solve for displacement increment
            dU_uu = system.solve(K_uu, R_uu)
            dU = np.zeros(len(R))
            dU[uu_idx] = dU_uu

            # Apply correction with optional line search
            integrator.update(model, dU)
            integrator.formUnbalance(model)

            if self._line_search:
                R_new = integrator.getUnbalance()
                norm_new = np.linalg.norm(R_new[uu_idx])

                if norm_new > norm_R:
                    # Full step didn't reduce residual — bisect
                    integrator.update(model, -dU)  # undo full step

                    alpha = 0.5
                    accepted = False
                    for ls in range(10):
                        integrator.update(model, alpha * dU)
                        integrator.formUnbalance(model)
                        R_ls = integrator.getUnbalance()
                        norm_ls = np.linalg.norm(R_ls[uu_idx])

                        if norm_ls < norm_R:
                            accepted = True
                            break

                        # Undo this trial and try smaller alpha
                        integrator.update(model, -alpha * dU)
                        alpha *= 0.5

                    if not accepted:
                        # Accept smallest alpha tried
                        integrator.update(model, alpha * dU)
                        integrator.formUnbalance(model)

        # If we exit the loop without converging
        raise RuntimeError(
            "Newton.solve(): failed to converge in {} iterations".format(test.maxIter))
