import numpy as np
from ..main import Integrator
from ...._systools.data import Vector, Matrix

class Newmark(Integrator):
    def __init__(self, ID=-1, gamma=0.5, beta=0.25):
        super().__init__(ID)
        self._gamma = gamma
        self._beta = beta

        # Committed state (from previous converged step)
        self._Ut = None  # displacement at t
        self._Vt = None  # velocity at t
        self._At = None  # acceleration at t

        # Trial state (current iteration)
        self._U = None
        self._V = None
        self._A = None

        # Newmark constants
        self._c1 = 1.0
        self._c2 = 0.0
        self._c3 = 0.0

        # Stored mass and damping (constant within a step)
        self._M_data = None
        self._C_data = None

    def _readCommittedState(self, model):
        """Read committed displacement, velocity, acceleration from nodes."""
        n = model.nDOF
        Ut = np.zeros(n)
        Vt = np.zeros(n)
        At = np.zeros(n)
        for node in model.nodes:
            node_dofs = node.getDOFs()
            if hasattr(node_dofs, 'tolist'):
                dof_list = node_dofs.tolist()
            else:
                dof_list = list(node_dofs)
            u_c = node._getCommitDisp()
            v_c = node._getCommitVel()
            a_c = node._getCommitAccel()
            for k, dof_idx in enumerate(dof_list):
                if k < len(u_c):
                    Ut[dof_idx] = u_c[k]
                if k < len(v_c):
                    Vt[dof_idx] = v_c[k]
                if k < len(a_c):
                    At[dof_idx] = a_c[k]
        return Ut, Vt, At

    def initialize(self, model):
        """Compute initial acceleration from M * a_0 = F_0 - K * u_0 - C * v_0."""
        self._Ut, self._Vt, self._At = self._readCommittedState(model)

        K_raw = np.asarray(model.K)
        M_raw = np.asarray(model.M)
        F_raw = np.asarray(model.F)
        C_raw = np.asarray(model.C) if model.C is not None else np.zeros_like(K_raw)

        rhs_a0 = F_raw - K_raw.dot(self._Ut) - C_raw.dot(self._Vt)

        M_diag = np.diag(M_raw)
        for i in range(len(self._At)):
            if abs(M_diag[i]) > 1e-30:
                self._At[i] = rhs_a0[i] / M_diag[i]
            else:
                self._At[i] = 0.0

        # Commit a_0 to nodes so newStep's _readCommittedState picks it up
        model.a = Vector(list(self._At))
        model._commit(model.F, model.u, model.v, model.a)

    def newStep(self, model, dt, current_time):
        """Set up the step: store committed state, compute predictor, store M/C."""
        super().newStep(model, dt, current_time)
        gamma = self._gamma
        beta = self._beta

        # Read committed state
        self._Ut, self._Vt, self._At = self._readCommittedState(model)

        # Newmark constants
        self._c1 = 1.0
        self._c2 = gamma / (beta * dt)
        self._c3 = 1.0 / (beta * dt * dt)

        # Store M, C (constant within step)
        self._M_data = np.asarray(model.M).copy()
        self._C_data = np.asarray(model.C).copy() if model.C is not None else np.zeros_like(self._M_data)

        # Predictor: a_pred, v_pred (u stays at Ut)
        a_pred = (-(1.0 / (beta * dt)) * self._Vt
                  - (1.0 / (2.0 * beta) - 1.0) * self._At)
        v_pred = (self._Vt
                  + dt * ((1.0 - gamma) * self._At + gamma * a_pred))

        # Initialize trial state at predictor
        self._U = self._Ut.copy()
        self._V = v_pred.copy()
        self._A = a_pred.copy()

        # Set on model and update domain
        model.u = Vector(list(self._U))
        model.v = Vector(list(self._V))
        model.a = Vector(list(self._A))
        model._commit(model.F, model.u, model.v, model.a)
        for element in model.elements:
            element._update()

    def formTangent(self, model, tangent='current'):
        """K_eff = c1*K_T + c2*C + c3*M."""
        if tangent == 'current':
            model._assemble(time=self._current_time + self._dt)
            # Restore u/v/a after assembly (assembly zeros u)
            model.u = Vector(list(self._U))
            model.v = Vector(list(self._V))
            model.a = Vector(list(self._A))
        K_T = np.asarray(model.K)
        self._K_eff = self._c1 * K_T + self._c2 * self._C_data + self._c3 * self._M_data

    def formUnbalance(self, model):
        """R = F_ext - F_int - M*a - C*v (equation of motion residual)."""
        F_ext = np.asarray(model.F)
        F_int = model.getInternalForce()
        self._R = (F_ext - F_int
                   - self._M_data.dot(self._A)
                   - self._C_data.dot(self._V))

    def update(self, model, dU):
        """U += dU, V += c2*dU, A += c3*dU; update domain."""
        self._U += dU
        self._V += self._c2 * dU
        self._A += self._c3 * dU

        model.u = Vector(list(self._U))
        model.v = Vector(list(self._V))
        model.a = Vector(list(self._A))
        model._commit(model.F, model.u, model.v, model.a)
        for element in model.elements:
            element._update()

    def commit(self, model):
        """Advance committed state after convergence."""
        self._Ut = self._U.copy()
        self._Vt = self._V.copy()
        self._At = self._A.copy()
        for element in model.elements:
            element._commit()
