import numpy as np
from ..main import Integrator
from ...._systools.data import Vector, Matrix

class CDiff(Integrator):
    """Central Difference explicit integrator following OpenSees architecture.

    Key properties:
    - Tangent = c3*M + c2*C (NO stiffness K)
    - "Garbage" v/a in newStep encode history terms u_{n-1}, u_n
    - update() receives FULL u_{n+1} (not increment)
    - Assembly at CURRENT time t (not t+dt)
    - Requires Linear algorithm (single solve per step)
    """
    def __init__(self, ID=-1):
        super().__init__(ID)
        self._Utm1 = None  # u at t - dt
        self._Ut = None    # u at t
        self._Vt = None    # v at t (for initial a computation)
        self._At = None    # a at t
        self._c2 = 0.0
        self._c3 = 0.0
        self._M_data = None
        self._C_data = None
        self._first_step = True

    def isExplicit(self):
        return True

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

    def _computeDamping(self, model, K_raw, M_raw):
        """Compute Rayleigh damping C = alphaM*M + betaK*K."""
        alphaM, betaK = model.getRayleighCoeffs()
        if alphaM != 0.0 or betaK != 0.0:
            return alphaM * M_raw + betaK * K_raw
        else:
            return np.zeros_like(M_raw)

    def initialize(self, model):
        """Compute initial acceleration and set Utm1.

        a_0 = M^{-1} * (F_0 - K*u_0 - C*v_0)
        u_{-1} = u_0 - dt*v_0 + 0.5*dt^2*a_0
        """
        self._Ut, self._Vt, self._At = self._readCommittedState(model)

        K_raw = self._system.getK().toarray()
        M_sp = self._system.getM()
        M_raw = M_sp.toarray() if M_sp is not None else np.zeros_like(K_raw)
        F_raw = np.asarray(model.F)
        C_raw = self._computeDamping(model, K_raw, M_raw)

        rhs_a0 = F_raw - K_raw.dot(self._Ut) - C_raw.dot(self._Vt)

        # Solve M*a0 = rhs using diagonal (exact for lumped mass)
        M_diag = np.diag(M_raw)
        for i in range(len(self._At)):
            if abs(M_diag[i]) > 1e-30:
                self._At[i] = rhs_a0[i] / M_diag[i]
            else:
                self._At[i] = 0.0

        # Commit a_0 to nodes so newStep's _readCommittedState picks it up
        model.a = Vector(list(self._At))
        model._commit(model.F, model.u, model.v, model.a)

        self._first_step = True

    def newStep(self, model, dt, current_time):
        """Set up the step with garbage v/a encoding history terms.

        Assembly happens at CURRENT time (not t+dt) for explicit.
        """
        super().newStep(model, dt, current_time)
        self._c2 = 0.5 / dt
        self._c3 = 1.0 / (dt * dt)

        # Store M, C from system
        K_data = self._system.getK().toarray()
        M_sp = self._system.getM()
        self._M_data = M_sp.toarray() if M_sp is not None else np.zeros_like(K_data)
        self._C_data = self._computeDamping(model, K_data, self._M_data)

        # Read committed state
        self._Ut, self._Vt, self._At = self._readCommittedState(model)

        if self._first_step:
            # Compute Utm1 from initial conditions
            self._Utm1 = (self._Ut - dt * self._Vt
                          + 0.5 * dt * dt * self._At)
            self._first_step = False

        # Set "garbage" v, a to encode history terms (OpenSees trick)
        # When plugged into standard residual R = F - F_int - M*a - C*v,
        # this produces the correct CD RHS automatically.
        v_garbage = -self._c2 * self._Utm1
        a_garbage = self._c3 * (self._Utm1 - 2.0 * self._Ut)

        model.v = Vector(list(v_garbage))
        model.a = Vector(list(a_garbage))
        model._commit(model.F, model.u, model.v, model.a)
        for element in model.elements:
            element._update()

    def formTangent(self, model, tangent='current'):
        """M_hat = c3*M + c2*C (no stiffness K for explicit)."""
        self._K_eff = self._c3 * self._M_data + self._c2 * self._C_data

    def formUnbalance(self, model):
        """R = F_ext - F_int - M*a_garbage - C*v_garbage.

        With garbage v/a from newStep, this produces the correct CD RHS.
        """
        F_ext = np.asarray(model.F)
        F_int = model.getInternalForce()
        v = np.asarray(model.v)
        a = np.asarray(model.a)
        self._R = F_ext - F_int - self._M_data.dot(a) - self._C_data.dot(v)

    def update(self, model, dU):
        """dU here is the full u_{n+1} (solved from M_hat * X = RHS).

        Compute v, a from central difference formulas and advance history.
        """
        dt = self._dt
        # For CD, the SOE solves M_hat * X = R where X is the FULL u_{n+1},
        # not an increment. The algorithm passes the raw SOE solution as dU.
        X = dU.copy()

        # Central difference velocity and acceleration
        v_new = self._c2 * (X - self._Utm1)            # v = (u_{n+1} - u_{n-1}) / (2*dt)
        a_new = self._c3 * (X - 2.0 * self._Ut + self._Utm1)  # a = (u_{n+1} - 2*u_n + u_{n-1}) / dt^2

        # Advance history
        self._Utm1 = self._Ut.copy()
        self._Ut = X.copy()
        self._Vt = v_new.copy()
        self._At = a_new.copy()

        # Set on model
        model.u = Vector(list(X))
        model.v = Vector(list(v_new))
        model.a = Vector(list(a_new))
        model._commit(model.F, model.u, model.v, model.a)
        for element in model.elements:
            element._update()

    def commit(self, model):
        """Commit elements after step."""
        for element in model.elements:
            element._commit()
