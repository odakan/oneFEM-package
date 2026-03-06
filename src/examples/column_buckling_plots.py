##-----------------------------------------------------------------------##
#                                                                         #
#   Column Buckling Validation — Visual Benchmark                         #
#   oneFEM P-Delta & Corotational Coordinate Transformations              #
#                                                                         #
#   Author: Onur Deniz Akan, IUSS Pavia                                   #
#   Date:   05/03/2026                                                    #
#                                                                         #
##-----------------------------------------------------------------------##
#
# PURPOSE
# -------
# This script visually validates the three geometric-nonlinearity
# formulations implemented in oneFEM (Linear, P-Delta, Corotational)
# by comparing FE results against closed-form analytical solutions for
# the classical Euler cantilever column buckling problem.
#
# The output is a 3×2 figure saved as column_buckling_validation.png:
#
#   Row 1  Load-amplification curves (P vs lateral tip displacement)
#          Compares FE results to the EXACT second-order elastic solution
#          for a cantilever beam-column under combined axial + lateral load.
#
#   Row 2  Linearized buckling analysis (eigenvalue tracking)
#          Monitors the minimum eigenvalue of the tangent stiffness K_uu
#          during displacement-controlled axial compression.  The load at
#          which this eigenvalue crosses zero is the linearized P_cr.
#
#   Row 3  Pushdown analysis (displacement-controlled post-buckling)
#          Pushes the column top down axially past P_cr with a small
#          constant lateral perturbation.  Tracks the axial reaction vs.
#          lateral displacement — the classic imperfect column equilibrium
#          path.
#
#
# PROBLEM SETUP
# -------------
# A vertical cantilever column of length L along the Y axis.
# Base (node 1) is fully fixed.  Tip (top node) is loaded.
#
#        tip  ←── F_lat (small lateral perturbation)
#         |        ↓ P (axial compression)
#         |
#         |   nElem beam-column elements
#         |
#        base ──── fixed (all DOFs)
#
# Properties are deliberately simple (not physical units) so that the
# analytical P_cr values are easy to verify by hand.
#
#
# ANALYTICAL REFERENCE
# --------------------
#
# 1) Euler buckling load for a cantilever (effective length K_eff = 2):
#
#        P_cr = pi^2 * E * I / (K_eff * L)^2 = pi^2 * E * I / (2L)^2
#
#    For the strong axis (I_z = 1):  P_cr_Iz = pi^2 * 1000 / 400 ≈ 24.674
#    For the weak   axis (I_y = 0.5): P_cr_Iy = pi^2 *  500 / 400 ≈ 12.337
#
# 2) P-Delta single-element cantilever P_cr:
#
#    A single cubic beam element with linearized geometric stiffness
#    (the P-Delta approximation) overestimates the Euler buckling load.
#    The analytical single-element P_cr for a cantilever is:
#
#        P_cr_pdelta_1 = 3 * E * I / L^2
#
#    This is 3/pi^2 * (2L)^2 / L^2 = 12/pi^2 ≈ 1.216 times the Euler value
#    (i.e., ~21.6% overestimate).  With more elements, P-Delta converges
#    toward the Euler value from above.
#
# 3) Exact second-order elastic amplification factor:
#
#    For a cantilever beam-column under simultaneous axial load P and
#    tip lateral load F, the exact lateral tip deflection is:
#
#        delta = delta_0 * phi(P)
#
#    where delta_0 = F * L^3 / (3 * E * I) is the linear tip deflection
#    (no axial load) and phi(P) is the amplification factor:
#
#        phi(P) = 3 * (tan(u) - u) / u^3
#        u      = (pi/2) * sqrt(P / P_cr)
#
#    This comes from solving the exact Euler-Bernoulli ODE with the
#    second-order moment P*y included:
#
#        E*I*y'' + P*y = F*(L - x)
#
#    IMPORTANT: This is NOT the "exact nonlinear" response.  It is the
#    exact LINEARIZED STABILITY solution (small rotations, small strains,
#    equilibrium written on the undeformed geometry but with the P*y
#    second-order moment term).  It is the appropriate reference for
#    validating P-Delta and Corotational formulations in the small-
#    displacement regime where all three should agree.
#
#    For truly large displacements (post-buckling), there is no simple
#    closed form.  The Corotational 10-element result serves as the
#    reference, validated by checking that P plateaus at the Euler P_cr.
#
#
# WHAT TO LOOK FOR IN THE PLOTS
# -----------------------------
#
# Row 1 (Amplification):
#   - Corot 10-elem (green) should nearly overlap the exact curve (black)
#   - P-Delta 1-elem (red) is stiffer — smaller deflections at the same P
#   - Corot 1-elem (blue) is also stiffer than 10-elem (insufficient DOFs)
#   - All curves diverge as P → P_cr (deflection → infinity)
#
# Row 2 (Eigenvalue):
#   - P-Delta 1-elem eigenvalue crosses zero at P ≈ 30 (= 3EI/L^2)
#   - Corot 10-elem eigenvalue crosses zero at P ≈ 24.67 (Euler value)
#   - Both eigenvalue curves are nearly linear (slight nonlinearity for Corot)
#
# Row 3 (Pushdown):
#   - Corot 10-elem P plateaus at the Euler P_cr horizontal line,
#     with lateral displacement growing — this is the imperfect column
#     equilibrium path approaching the bifurcation load from below
#   - P-Delta 1-elem P asymptotes to 3EI/L^2 (its own higher P_cr)
#   - Corot 1-elem overshoots past P_cr because a single cubic element
#     overestimates the buckling stiffness
#   - The P-Delta formulation CANNOT capture the true post-buckling path
#     (it uses linearized geometry) — it just sees P growing until it
#     hits its own (overestimated) limit
#

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — write to file, no GUI
import matplotlib.pyplot as plt
import sys, os

# Add parent directory to path so we can import oneFEM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from oneFEM.model import Domain
from oneFEM.model.element.beam import ElasticBeamColumn2d, ElasticBeamColumn3d
from oneFEM.model.element.kinematics.crdTransf import (
    LinearCrdTransf2d, PDeltaCrdTransf2d, CorotCrdTransf2d,
    PDeltaCrdTransf3d, CorotCrdTransf3d
)
from oneFEM.model.node import Node23, Node36
from oneFEM.model.tseries import Linear as LinearTS, Constant as ConstantTS
from oneFEM.model.pattern import Plain as PlainPattern
from oneFEM.analysis import Analysis
from oneFEM.analysis.algorithm.newton_raphson import Newton
from oneFEM.analysis.constraints import Plain as PlainConstraints
from oneFEM.analysis.numberer import Plain as PlainNumberer
from oneFEM.analysis.system import FullGeneral
from oneFEM.analysis.integrator import LoadControl, DispControl
from oneFEM.analysis.test import NormUnbalance
from oneFEM import SimulationManager
from oneFEM._systools.data import Vector


# ================================================================
#  Material and section properties
# ================================================================
# Deliberately simple values — not physical units — so that analytical
# P_cr can be computed by hand.
#
#   E  = 1000   Young's modulus
#   A  = 1      Cross-section area
#   Iz = 1      Strong-axis second moment of area (bending about Z)
#   Iy = 0.5    Weak-axis second moment of area (bending about Y)
#   G  = 400    Shear modulus (3D only — needed for ElasticBeamColumn3d)
#   J  = 0.8    Torsional constant (3D only)
#   L  = 10     Column length
#   F_lat = 0.01  Small lateral tip load (for load-amplification tests)

E = 1000.0;  A = 1.0;  Iz = 1.0;  Iy = 0.5
G = 400.0;   J = 0.8;  L = 10.0;  F_lat = 0.01


# ================================================================
#  Analytical reference values
# ================================================================

# Euler buckling load for cantilever (effective length factor K=2):
#   P_cr = pi^2 * E * I / (2*L)^2
P_cr_euler_Iz = np.pi**2 * E * Iz / (2*L)**2   # ≈ 24.674 (strong axis)
P_cr_euler_Iy = np.pi**2 * E * Iy / (2*L)**2   # ≈ 12.337 (weak axis)

# P-Delta single-element cantilever P_cr:
#   P_cr = 3 * E * I / L^2
# This comes from the determinant of the condensed 2×2 stiffness matrix
# (lateral DOF + rotation at the tip) with the geometric stiffness included.
# A single cubic element can only represent a linear lateral displacement
# shape, which overestimates the true sinusoidal buckling mode.
P_cr_pdelta_1_Iz = 3*E*Iz / L**2   # = 30.0 (21.6% above Euler)
P_cr_pdelta_1_Iy = 3*E*Iy / L**2   # = 15.0 (21.6% above Euler)


def exact_amplification(P, P_cr):
    """
    Exact second-order elastic amplification factor for a cantilever
    beam-column under simultaneous axial load P and tip lateral load F.

    The tip deflection is:  delta = delta_0 * phi(P)
    where delta_0 = F*L^3 / (3*E*I) is the first-order deflection.

    This function returns phi(P):

        phi = 3 * (tan(u) - u) / u^3

    where u = (pi/2) * sqrt(P / P_cr).

    Derivation: Solve the Euler-Bernoulli ODE  E*I*y'' + P*y = F*(L-x)
    with boundary conditions y(0) = 0, y'(0) = 0 (cantilever base).
    The general solution involves sin/cos terms; evaluating at x = L
    gives the closed-form tip deflection.

    Reference: Timoshenko & Gere, "Theory of Elastic Stability", Ch. 1.
               Also: Bazant & Cedolin, "Stability of Structures", Ch. 2.

    NOTE: This is NOT a truly nonlinear result.  It assumes small
    rotations (sin(theta) ~ theta) and equilibrium on the original
    geometry with only the second-order moment P*y added.  For large
    displacements, no simple closed form exists.

    :param P:     Axial compression force (positive = compression)
    :param P_cr:  Euler buckling load for this column configuration
    :return:      Amplification factor phi >= 1.0 (or inf if P >= P_cr)
    """
    ratio = P / P_cr
    if ratio >= 1.0:
        return float('inf')
    u = (np.pi / 2.0) * np.sqrt(ratio)
    if abs(u) < 1e-10:
        return 1.0
    return 3.0 * (np.tan(u) - u) / u**3


# ================================================================
#  Helper functions — build models and run analyses
# ================================================================
# Each helper builds a fresh model from scratch for each test case.
# This is intentional: we want fully independent models for each
# load level / formulation, with no state leaking between tests.


def amplification_curve_2d(TransfClass, nElem, P_levels):
    """
    Load-controlled amplification test (2D).

    For each axial load level P in P_levels, build a cantilever column,
    apply P (axial, downward) + F_lat (lateral, horizontal) simultaneously,
    and solve with Newton-Raphson.  Return the lateral tip displacement
    at each load level.

    The loads are applied via a Linear time series over 10 steps of dt=0.1,
    so the full load is reached at time = 1.0.  Newton with current tangent
    ensures geometric nonlinearity is captured iteratively.

    Model:
        - nElem beam-column elements along Y axis
        - Node23 (2D, 3-DOF: ux, uy, rz)
        - Base fully fixed, tip free
        - CrdTransf from TransfClass (PDelta or Corot)

    :param TransfClass:  Coordinate transformation class
    :param nElem:        Number of elements along the column
    :param P_levels:     Array of axial load values to test
    :return:             Array of lateral tip displacements (same length as P_levels)
    """
    deltas = []
    for P_load in P_levels:
        dL_elem = L / nElem
        model = Domain(nD=2)
        nodes = []
        for i in range(nElem + 1):
            y = i * dL_elem
            fix = [True, True, True] if i == 0 else None
            nd = Node23(i + 1, coord=[0.0, y], fix=fix)
            nodes.append(nd); model.add(nd)
        for i in range(nElem):
            tr = TransfClass()
            b = ElasticBeamColumn2d(i+1, nodes=[nodes[i], nodes[i+1]],
                                    A=A, E=E, I=Iz, transf=tr)
            model.add(b)
        # Linear time series: load scales from 0 to 1.0 over the analysis
        ts = LinearTS(1, factor=1.0)
        # Combined load: F_lat in X + P_load compression in -Y
        pat = PlainPattern(1, tseries=ts,
                           load=[[nElem+1, F_lat, -P_load, 0.0]])
        model.add(pat)
        # Newton with current tangent — essential for geometric nonlinearity
        alg = Newton(1, tangent='current')
        integ = LoadControl(1)
        ctest = NormUnbalance(1, tol=1e-10, maxIter=80)
        an = Analysis(1, algorithm=alg, constraints=PlainConstraints(1),
                      integrator=integ, system=FullGeneral(1), test=ctest)
        sim = SimulationManager(1, model, an, dt=0.1)
        try:
            sim.analyze(10, dt=0.1)
            u = nodes[-1]._getCommitDisp()
            deltas.append(float(u[0]))  # lateral = X direction
        except RuntimeError:
            deltas.append(np.nan)
    return np.array(deltas)


def amplification_curve_3d(TransfClass, nElem, P_levels, vecxz):
    """
    Load-controlled amplification test (3D, weak-axis bending).

    Same concept as amplification_curve_2d, but in 3D.  The lateral
    perturbation is applied in the Z direction (weak axis, I_y = 0.5)
    so that buckling occurs about the weak axis.

    Model:
        - nElem beam-column elements along Y axis
        - Node36 (3D, 6-DOF: ux, uy, uz, rx, ry, rz)
        - vecxz = [0,0,1] defines the local xz-plane orientation
        - Base fully fixed (all 6 DOFs), tip free

    :param TransfClass:  3D coordinate transformation class
    :param nElem:        Number of elements
    :param P_levels:     Array of axial load values
    :param vecxz:        Vector in the local xz-plane (for 3D orientation)
    :return:             Array of lateral tip displacements in Z
    """
    deltas = []
    for P_load in P_levels:
        dL_elem = L / nElem
        model = Domain(nD=3)
        nodes = []
        for i in range(nElem + 1):
            y = i * dL_elem
            fix = [True]*6 if i == 0 else None
            nd = Node36(i+1, coord=[0.0, y, 0.0], fix=fix)
            nodes.append(nd); model.add(nd)
        for i in range(nElem):
            tr = TransfClass(vecxz=vecxz)
            b = ElasticBeamColumn3d(i+1, nodes=[nodes[i], nodes[i+1]],
                                    A=A, E=E, Iz=Iz, Iy=Iy, G=G, J=J,
                                    transf=tr)
            model.add(b)
        ts = LinearTS(1, factor=1.0)
        # Load: compression in -Y, perturbation in +Z (weak axis)
        pat = PlainPattern(1, tseries=ts,
                           load=[[nElem+1, 0.0, -P_load, F_lat, 0.0, 0.0, 0.0]])
        model.add(pat)
        alg = Newton(1, tangent='current')
        integ = LoadControl(1)
        ctest = NormUnbalance(1, tol=1e-10, maxIter=80)
        an = Analysis(1, algorithm=alg, constraints=PlainConstraints(1),
                      integrator=integ, system=FullGeneral(1), test=ctest)
        sim = SimulationManager(1, model, an, dt=0.1)
        try:
            sim.analyze(10, dt=0.1)
            u = nodes[-1]._getCommitDisp()
            deltas.append(float(u[2]))  # lateral = Z direction
        except RuntimeError:
            deltas.append(np.nan)
    return np.array(deltas)


def eigenvalue_curve(TransfClass_2d, nElem, incr, nSteps, dim='2d',
                     TransfClass_3d=None, vecxz=None):
    """
    Linearized buckling analysis via eigenvalue tracking.

    Uses displacement-controlled axial compression (DisplacementControl
    integrator) to push the column top downward incrementally.  At each
    step, after converging the equilibrium, reassembles the tangent
    stiffness K and extracts the minimum eigenvalue of K_uu (the free-DOF
    partition).

    The minimum eigenvalue decreases as axial load increases.  When it
    crosses zero, the tangent stiffness is singular — this is the
    linearized buckling load P_cr.  This is the standard approach for
    detecting bifurcation buckling in FE analysis.

    Why eigenvalue tracking instead of convergence failure?
    Euler buckling is a BIFURCATION — the straight (trivial) equilibrium
    path always exists, even past P_cr.  With no perturbation, the
    Corotational formulation will happily follow the straight path
    indefinitely.  Convergence failure only occurs at LIMIT POINTS
    (turning points in the equilibrium path), not at bifurcations.
    Eigenvalue tracking detects the bifurcation directly.

    Setup:
        - Top node: vertical DOF (Y) is FIXED — this is the displacement-
          controlled DOF.  The DispControl integrator prescribes a
          negative displacement increment (compression) at each step.
        - No external load — pure compression via prescribed displacement.
        - At each step, we extract the axial reaction from the internal
          force at the base node.

    :param TransfClass_2d:  2D CrdTransf class (or None if dim='3d')
    :param nElem:           Number of elements
    :param incr:            Displacement increment per step (negative = compression)
    :param nSteps:          Maximum number of analysis steps
    :param dim:             '2d' or '3d'
    :param TransfClass_3d:  3D CrdTransf class (required if dim='3d')
    :param vecxz:           Local xz-plane vector (required if dim='3d')
    :return:                (P_array, min_eigenvalue_array) — one entry per step
    """
    dL_elem = L / nElem
    if dim == '2d':
        model = Domain(nD=2)
        nodes = []
        for i in range(nElem + 1):
            y = i * dL_elem
            if i == 0:
                # Base: fully fixed (ux, uy, rz)
                fix = [True, True, True]
            elif i == nElem:
                # Top: fix uy (vertical) for displacement control,
                #      leave ux and rz free
                fix = [False, True, False]
            else:
                fix = None
            nd = Node23(i+1, coord=[0.0, y], fix=fix)
            nodes.append(nd); model.add(nd)
        for i in range(nElem):
            tr = TransfClass_2d()
            b = ElasticBeamColumn2d(i+1, nodes=[nodes[i], nodes[i+1]],
                                    A=A, E=E, I=Iz, transf=tr)
            model.add(b)
    else:
        model = Domain(nD=3)
        nodes = []
        for i in range(nElem + 1):
            y = i * dL_elem
            if i == 0:
                fix = [True]*6
            elif i == nElem:
                # Top: fix uy only (DOF index 1)
                fix = [False, True, False, False, False, False]
            else:
                fix = None
            nd = Node36(i+1, coord=[0.0, y, 0.0], fix=fix)
            nodes.append(nd); model.add(nd)
        for i in range(nElem):
            tr = TransfClass_3d(vecxz=vecxz)
            b = ElasticBeamColumn3d(i+1, nodes=[nodes[i], nodes[i+1]],
                                    A=A, E=E, Iz=Iz, Iy=Iy, G=G, J=J,
                                    transf=tr)
            model.add(b)

    # Displacement control at the top node, DOF 1 (vertical/Y)
    top_node = nodes[-1]
    alg = Newton(1, tangent='current')
    integ = DispControl(node=top_node, dof=1, incr=incr)
    ctest = NormUnbalance(1, tol=1e-8, maxIter=50)
    an = Analysis(1, algorithm=alg, constraints=PlainConstraints(1),
                  integrator=integ, system=FullGeneral(1), test=ctest)

    # Initialize DOF numbering and partition into free (uu) / fixed (pp)
    model._domain(); an._organize(model)
    uu_idx = np.array(an.uu, dtype=int)  # free DOF indices

    # Get the base node's global DOF indices (for extracting reactions)
    base_dofs = nodes[0].getDOFs()
    if hasattr(base_dofs, 'tolist'):
        bd = base_dofs.tolist()
    elif hasattr(base_dofs, 'data'):
        bd = base_dofs.data.tolist()
    else:
        bd = list(base_dofs)

    P_arr, eig_arr = [], []
    for step in range(nSteps):
        try:
            # 1. Assemble tangent stiffness K and external force F
            model._assemble(time=an._time + 1.0)
            # 2. Apply displacement increment at controlled DOF
            integ.newStep(model, 1.0, an._time)
            # 3. Newton-Raphson iteration to equilibrium
            alg.solve(model, an.uu, an.pp, integ, FullGeneral(1), ctest)
            # 4. Commit converged state
            integ.commit(model)
            an._time += 1.0

            # Extract axial reaction from internal force at base
            # bd[1] is the base node's Y-direction global DOF
            F_int = model.getInternalForce()
            P_arr.append(abs(F_int[bd[1]]))

            # Reassemble to get the tangent stiffness at the converged state
            model._assemble(time=an._time)
            K = np.asarray(model.K)
            # Extract the free-DOF partition K_uu
            K_uu = K[np.ix_(uu_idx, uu_idx)]
            # Compute all eigenvalues (symmetric matrix → eigvalsh)
            eigs = np.linalg.eigvalsh(K_uu)
            # Track the minimum eigenvalue — when it crosses zero, P = P_cr
            eig_arr.append(np.min(eigs))
        except RuntimeError:
            break
    return np.array(P_arr), np.array(eig_arr)


def get_deformed_shape_2d(TransfClass, nElem, P_target):
    """
    Compute deformed column shape at a given axial load level (2D).

    Used for visualization — returns the (x, y) coordinates of every
    node in the deformed configuration.

    :param TransfClass:  2D CrdTransf class
    :param nElem:        Number of elements
    :param P_target:     Target axial load (with F_lat lateral perturbation)
    :return:             (x_array, y_array) of deformed node positions
    """
    dL_elem = L / nElem
    model = Domain(nD=2)
    nodes = []
    for i in range(nElem + 1):
        y = i * dL_elem
        fix = [True, True, True] if i == 0 else None
        nd = Node23(i+1, coord=[0.0, y], fix=fix)
        nodes.append(nd); model.add(nd)
    for i in range(nElem):
        tr = TransfClass()
        b = ElasticBeamColumn2d(i+1, nodes=[nodes[i], nodes[i+1]],
                                A=A, E=E, I=Iz, transf=tr)
        model.add(b)
    ts = LinearTS(1, factor=1.0)
    pat = PlainPattern(1, tseries=ts,
                       load=[[nElem+1, F_lat, -P_target, 0.0]])
    model.add(pat)
    alg = Newton(1, tangent='current')
    integ = LoadControl(1)
    ctest = NormUnbalance(1, tol=1e-10, maxIter=80)
    an = Analysis(1, algorithm=alg, constraints=PlainConstraints(1),
                  integrator=integ, system=FullGeneral(1), test=ctest)
    sim = SimulationManager(1, model, an, dt=0.1)
    # Use 20 steps of dt=0.05 for smoother load application
    sim.analyze(20, dt=0.05)

    # Extract deformed positions: original coord + displacement
    xs, ys = [], []
    for nd in nodes:
        c = nd._coord
        u = nd._getCommitDisp()
        xs.append(float(c[0]) + float(u[0]))
        ys.append(float(c[1]) + float(u[1]))
    return np.array(xs), np.array(ys)


def pushdown_2d(TransfClass, nElem, incr, nSteps, F_perturb=0.01):
    """
    Displacement-controlled pushdown analysis (2D).

    This is the key test for geometric nonlinearity: push the column top
    downward via displacement control while applying a small CONSTANT
    lateral perturbation force.  Track the axial reaction and lateral
    displacement at every step.

    The perturbation breaks the symmetry of the perfect column, turning
    the bifurcation into a smooth equilibrium path.  As the axial
    reaction approaches P_cr, the lateral displacement grows rapidly —
    this is the classic imperfect column response.

    Key setup details:
        - The perturbation uses a CONSTANT time series (not Linear),
          so it stays at F_perturb for all steps.  A Linear time series
          would grow the perturbation proportionally with pseudo-time,
          artificially increasing the lateral load and corrupting the
          P_cr estimate.
        - The top node's vertical DOF is FIXED (in the pp partition)
          so that DispControl can prescribe its displacement.
        - Newton-Raphson only solves for the remaining free DOFs
          (lateral + rotation at each node).

    Why P-Delta blows up:
        P-Delta uses a linearized geometric stiffness K_g = (N/L)*outer(r,r)
        that is only valid for small deformations.  Near P_cr, the tangent
        stiffness becomes nearly singular, and the linearized correction
        produces huge lateral displacements that violate the small-angle
        assumption.  The divergence guard (lat > 100*L) catches this.

    Why Corot works past P_cr:
        The Corotational formulation continuously updates the local frame
        to follow the deformed element chord.  It correctly captures the
        geometry change at large displacements, so the equilibrium path
        remains stable as P approaches and plateaus at P_cr.

    :param TransfClass:  2D CrdTransf class (PDelta or Corot)
    :param nElem:        Number of elements
    :param incr:         Axial displacement increment per step (negative = compression)
    :param nSteps:       Maximum number of analysis steps
    :param F_perturb:    Constant lateral perturbation force magnitude
    :return:             (P_arr, delta_lat_arr, delta_axial_arr)
    """
    dL_elem = L / nElem
    model = Domain(nD=2)
    nodes = []
    for i in range(nElem + 1):
        y = i * dL_elem
        if i == 0:
            # Base: fully fixed
            fix = [True, True, True]
        elif i == nElem:
            # Top: fix vertical DOF (index 1) for displacement control
            fix = [False, True, False]
        else:
            fix = None
        nd = Node23(i + 1, coord=[0.0, y], fix=fix)
        nodes.append(nd); model.add(nd)
    for i in range(nElem):
        tr = TransfClass()
        b = ElasticBeamColumn2d(i+1, nodes=[nodes[i], nodes[i+1]],
                                A=A, E=E, I=Iz, transf=tr)
        model.add(b)
    # CONSTANT time series — perturbation does NOT grow with pseudo-time
    ts = ConstantTS(1, factor=1.0)
    pat = PlainPattern(1, tseries=ts,
                       load=[[nElem+1, F_perturb, 0.0, 0.0]])
    model.add(pat)

    top_node = nodes[-1]
    alg = Newton(1, tangent='current')
    integ = DispControl(node=top_node, dof=1, incr=incr)
    ctest = NormUnbalance(1, tol=1e-8, maxIter=50)
    an = Analysis(1, algorithm=alg, constraints=PlainConstraints(1),
                  integrator=integ, system=FullGeneral(1), test=ctest)

    model._domain(); an._organize(model)

    # Get base node global DOF indices for extracting axial reaction
    base_dofs = nodes[0].getDOFs()
    if hasattr(base_dofs, 'tolist'):
        bd = base_dofs.tolist()
    elif hasattr(base_dofs, 'data'):
        bd = base_dofs.data.tolist()
    else:
        bd = list(base_dofs)

    P_arr, dlat_arr, dax_arr = [], [], []
    for step in range(nSteps):
        try:
            model._assemble(time=an._time + 1.0)
            integ.newStep(model, 1.0, an._time)
            alg.solve(model, an.uu, an.pp, integ, FullGeneral(1), ctest)
            integ.commit(model)
            an._time += 1.0

            # Axial reaction = internal force at base Y-DOF
            F_int = model.getInternalForce()
            P_arr.append(abs(F_int[bd[1]]))

            # Tip displacements
            u_top = top_node._getCommitDisp()
            lat = abs(float(u_top[0]))   # lateral = X
            dlat_arr.append(lat)
            dax_arr.append(abs(float(u_top[1])))  # axial = Y

            # Divergence guard: P-Delta formulation produces nonsensical
            # lateral displacements near/past its P_cr.  Stop collecting
            # data when deflections exceed 100× column length.
            if lat > 100 * L:
                break
        except RuntimeError:
            break
    return np.array(P_arr), np.array(dlat_arr), np.array(dax_arr)


def pushdown_3d(TransfClass, nElem, incr, nSteps, vecxz, F_perturb=0.01):
    """
    Displacement-controlled pushdown analysis (3D, weak-axis perturbation).

    Same concept as pushdown_2d but in 3D.  The lateral perturbation is
    applied in the Z direction (weak axis, I_y = 0.5) to trigger
    weak-axis buckling.  The expected P_cr plateau is at the weak-axis
    Euler value ≈ 12.337.

    :param TransfClass:  3D CrdTransf class (PDelta or Corot)
    :param nElem:        Number of elements
    :param incr:         Axial displacement increment per step (negative)
    :param nSteps:       Maximum number of analysis steps
    :param vecxz:        Local xz-plane vector for 3D orientation
    :param F_perturb:    Constant lateral perturbation force in Z
    :return:             (P_arr, delta_lat_arr, delta_axial_arr)
    """
    dL_elem = L / nElem
    model = Domain(nD=3)
    nodes = []
    for i in range(nElem + 1):
        y = i * dL_elem
        if i == 0:
            fix = [True]*6
        elif i == nElem:
            # Top: fix vertical DOF (index 1) for displacement control
            fix = [False, True, False, False, False, False]
        else:
            fix = None
        nd = Node36(i+1, coord=[0.0, y, 0.0], fix=fix)
        nodes.append(nd); model.add(nd)
    for i in range(nElem):
        tr = TransfClass(vecxz=vecxz)
        b = ElasticBeamColumn3d(i+1, nodes=[nodes[i], nodes[i+1]],
                                A=A, E=E, Iz=Iz, Iy=Iy, G=G, J=J,
                                transf=tr)
        model.add(b)
    # Constant perturbation in Z (weak axis) — does NOT grow with time
    ts = ConstantTS(1, factor=1.0)
    pat = PlainPattern(1, tseries=ts,
                       load=[[nElem+1, 0.0, 0.0, F_perturb, 0.0, 0.0, 0.0]])
    model.add(pat)

    top_node = nodes[-1]
    alg = Newton(1, tangent='current')
    integ = DispControl(node=top_node, dof=1, incr=incr)
    ctest = NormUnbalance(1, tol=1e-8, maxIter=50)
    an = Analysis(1, algorithm=alg, constraints=PlainConstraints(1),
                  integrator=integ, system=FullGeneral(1), test=ctest)

    model._domain(); an._organize(model)
    base_dofs = nodes[0].getDOFs()
    if hasattr(base_dofs, 'tolist'):
        bd = base_dofs.tolist()
    elif hasattr(base_dofs, 'data'):
        bd = base_dofs.data.tolist()
    else:
        bd = list(base_dofs)

    P_arr, dlat_arr, dax_arr = [], [], []
    for step in range(nSteps):
        try:
            model._assemble(time=an._time + 1.0)
            integ.newStep(model, 1.0, an._time)
            alg.solve(model, an.uu, an.pp, integ, FullGeneral(1), ctest)
            integ.commit(model)
            an._time += 1.0

            F_int = model.getInternalForce()
            P_arr.append(abs(F_int[bd[1]]))

            u_top = top_node._getCommitDisp()
            lat = abs(float(u_top[2]))  # Z = weak axis lateral displacement
            dlat_arr.append(lat)
            dax_arr.append(abs(float(u_top[1])))
            if lat > 100 * L:  # divergence guard
                break
        except RuntimeError:
            break
    return np.array(P_arr), np.array(dlat_arr), np.array(dax_arr)


# ================================================================
#  Run all analyses
# ================================================================

# --- Row 1: Load-amplification curves ---
# Sweep axial load from 0 to 0.9*P_cr in 30 steps.
# At each level, solve for the lateral tip displacement.
# Compare P-Delta 1-elem, Corot 1-elem, Corot 10-elem vs analytical.

print("Computing 2D amplification curves...")
P_levels_2d = np.linspace(0, 0.9 * P_cr_euler_Iz, 30)
# First-order (no axial load) lateral tip deflection of cantilever:
#   delta_0 = F * L^3 / (3 * E * I)
delta_0_2d = F_lat * L**3 / (3 * E * Iz)

d_pdelta_1  = amplification_curve_2d(PDeltaCrdTransf2d, 1,  P_levels_2d)
d_corot_1   = amplification_curve_2d(CorotCrdTransf2d,  1,  P_levels_2d)
d_corot_10  = amplification_curve_2d(CorotCrdTransf2d,  10, P_levels_2d)
# Analytical: delta = delta_0 * phi(P)
d_exact_2d  = np.array([delta_0_2d * exact_amplification(P, P_cr_euler_Iz)
                         for P in P_levels_2d])

print("Computing 3D amplification curves (weak axis)...")
P_levels_3d = np.linspace(0, 0.9 * P_cr_euler_Iy, 30)
delta_0_3d = F_lat * L**3 / (3 * E * Iy)
vxz = [0, 0, 1]  # local xz-plane: Z direction

d3_pdelta_1  = amplification_curve_3d(PDeltaCrdTransf3d, 1,  P_levels_3d, vxz)
d3_corot_1   = amplification_curve_3d(CorotCrdTransf3d,  1,  P_levels_3d, vxz)
d3_corot_10  = amplification_curve_3d(CorotCrdTransf3d,  10, P_levels_3d, vxz)
d3_exact     = np.array([delta_0_3d * exact_amplification(P, P_cr_euler_Iy)
                         for P in P_levels_3d])

# --- Row 2: Eigenvalue tracking ---
# Displacement-controlled compression with incr = -0.002 per step (no lateral load).
# At each step, extract min eigenvalue of K_uu.

print("Computing 2D eigenvalue curves...")
P_pd1_2d,  eig_pd1_2d  = eigenvalue_curve(PDeltaCrdTransf2d, 1,  -0.002, 200, '2d')
P_cor10_2d, eig_cor10_2d = eigenvalue_curve(CorotCrdTransf2d,  10, -0.002, 200, '2d')

print("Computing 3D eigenvalue curves (weak axis)...")
P_pd1_3d,  eig_pd1_3d  = eigenvalue_curve(None, 1,  -0.002, 150, '3d',
                                            PDeltaCrdTransf3d, vxz)
P_cor10_3d, eig_cor10_3d = eigenvalue_curve(None, 10, -0.002, 150, '3d',
                                             CorotCrdTransf3d, vxz)

# --- Row 3: Pushdown analysis ---
# Displacement-controlled compression (incr = -0.002) + constant F_perturb = 0.01.
# Track axial reaction P and lateral tip displacement at each step.
# The P vs delta_lat curve shows the imperfect column equilibrium path.

print("Computing 2D pushdown (displacement-controlled compression)...")
pd_P_pd1_2d, pd_dlat_pd1_2d, pd_dax_pd1_2d = pushdown_2d(
    PDeltaCrdTransf2d, 1, -0.002, 250)
pd_P_cor1_2d, pd_dlat_cor1_2d, pd_dax_cor1_2d = pushdown_2d(
    CorotCrdTransf2d, 1, -0.002, 250)
pd_P_cor10_2d, pd_dlat_cor10_2d, pd_dax_cor10_2d = pushdown_2d(
    CorotCrdTransf2d, 10, -0.002, 250)

print("Computing 3D pushdown (weak-axis compression)...")
pd_P_pd1_3d, pd_dlat_pd1_3d, pd_dax_pd1_3d = pushdown_3d(
    PDeltaCrdTransf3d, 1, -0.002, 200, vxz)
pd_P_cor1_3d, pd_dlat_cor1_3d, pd_dax_cor1_3d = pushdown_3d(
    CorotCrdTransf3d, 1, -0.002, 200, vxz)
pd_P_cor10_3d, pd_dlat_cor10_3d, pd_dax_cor10_3d = pushdown_3d(
    CorotCrdTransf3d, 10, -0.002, 200, vxz)


# ================================================================
#  Plotting — 3 rows × 2 columns
# ================================================================
# Left column:  2D (strong axis, Iz)
# Right column: 3D (weak axis, Iy)

fig, axes = plt.subplots(3, 2, figsize=(14, 16))
fig.suptitle('Column Buckling Validation — oneFEM PDelta & Corotational CrdTransf',
             fontsize=14, fontweight='bold')

# ---- Row 1, Col 1: 2D Load-Amplification ----
# Compare FE lateral tip displacement vs exact beam-column formula.
# X-axis: lateral displacement (scaled ×1e3 for readability)
# Y-axis: axial force P
# Expect: Corot 10-elem ≈ exact, P-Delta stiffer, all diverge near P_cr.
ax = axes[0, 0]
ax.plot(d_exact_2d * 1e3, P_levels_2d, 'k-', lw=2, label='Exact (beam-column)')
ax.plot(d_pdelta_1 * 1e3, P_levels_2d, 'r--', lw=1.5, label='P-Delta 1-elem')
ax.plot(d_corot_1 * 1e3, P_levels_2d, 'b:', lw=1.5, label='Corot 1-elem')
ax.plot(d_corot_10 * 1e3, P_levels_2d, 'g-.', lw=2, label='Corot 10-elem')
ax.axhline(P_cr_euler_Iz, color='grey', ls='--', alpha=0.5)
ax.text(0.02, P_cr_euler_Iz + 0.3, f'$P_{{cr}}$={P_cr_euler_Iz:.2f}',
        fontsize=8, color='grey')
ax.set_xlabel('Lateral tip displacement (×$10^{-3}$)')
ax.set_ylabel('Axial force P')
ax.set_title('2D: Load-amplification (strong axis $I_z$)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ---- Row 1, Col 2: 3D Load-Amplification (weak axis) ----
# Same as 2D but for weak-axis bending (Iy = 0.5, lower P_cr).
ax = axes[0, 1]
ax.plot(d3_exact * 1e3, P_levels_3d, 'k-', lw=2, label='Exact (beam-column)')
ax.plot(d3_pdelta_1 * 1e3, P_levels_3d, 'r--', lw=1.5, label='P-Delta 1-elem')
ax.plot(d3_corot_1 * 1e3, P_levels_3d, 'b:', lw=1.5, label='Corot 1-elem')
ax.plot(d3_corot_10 * 1e3, P_levels_3d, 'g-.', lw=2, label='Corot 10-elem')
ax.axhline(P_cr_euler_Iy, color='grey', ls='--', alpha=0.5)
ax.text(0.02, P_cr_euler_Iy + 0.15, f'$P_{{cr}}$={P_cr_euler_Iy:.2f}',
        fontsize=8, color='grey')
ax.set_xlabel('Lateral tip displacement Z (×$10^{-3}$)')
ax.set_ylabel('Axial force P')
ax.set_title('3D: Load-amplification (weak axis $I_y$)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ---- Row 2, Col 1: 2D Eigenvalue Tracking ----
# Min eigenvalue of K_uu vs axial force P.
# Expect: eigenvalue drops to zero at the respective P_cr for each formulation.
# Vertical reference lines mark the analytical P_cr values.
ax = axes[1, 0]
ax.plot(P_pd1_2d, eig_pd1_2d, 'r-', lw=1.5, label='P-Delta 1-elem')
ax.plot(P_cor10_2d, eig_cor10_2d, 'g-', lw=1.5, label='Corot 10-elem')
ax.axhline(0, color='k', ls='-', lw=0.5)
ax.axvline(P_cr_euler_Iz, color='grey', ls='--', alpha=0.5,
           label=f'Euler $P_{{cr}}$={P_cr_euler_Iz:.2f}')
ax.axvline(P_cr_pdelta_1_Iz, color='salmon', ls=':', alpha=0.5,
           label=f'P-Delta(1) $P_{{cr}}$={P_cr_pdelta_1_Iz:.1f}')
ax.set_xlabel('Axial force P')
ax.set_ylabel('Min eigenvalue of $K_{uu}$')
ax.set_title('2D: Tangent stiffness eigenvalue vs P')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# ---- Row 2, Col 2: 3D Eigenvalue Tracking (weak axis) ----
ax = axes[1, 1]
ax.plot(P_pd1_3d, eig_pd1_3d, 'r-', lw=1.5, label='P-Delta 1-elem')
ax.plot(P_cor10_3d, eig_cor10_3d, 'g-', lw=1.5, label='Corot 10-elem')
ax.axhline(0, color='k', ls='-', lw=0.5)
ax.axvline(P_cr_euler_Iy, color='grey', ls='--', alpha=0.5,
           label=f'Euler $P_{{cr}}$={P_cr_euler_Iy:.2f}')
ax.axvline(P_cr_pdelta_1_Iy, color='salmon', ls=':', alpha=0.5,
           label=f'P-Delta(1) $P_{{cr}}$={P_cr_pdelta_1_Iy:.1f}')
ax.set_xlabel('Axial force P')
ax.set_ylabel('Min eigenvalue of $K_{uu}$')
ax.set_title('3D: Weak-axis tangent eigenvalue vs P')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# ---- Row 3, Col 1: 2D Pushdown ----
# Axial reaction P vs lateral tip displacement — the imperfect column
# equilibrium path under displacement-controlled compression.
# Expect:
#   - Corot 10-elem (green) plateaus at Euler P_cr ≈ 24.67
#   - P-Delta 1-elem (red) asymptotes to 3EI/L² = 30.0
#   - Corot 1-elem (blue) overshoots past P_cr (too few DOFs)
ax = axes[2, 0]
ax.plot(pd_dlat_pd1_2d, pd_P_pd1_2d, 'r--', lw=1.5, label='P-Delta 1-elem')
ax.plot(pd_dlat_cor1_2d, pd_P_cor1_2d, 'b:', lw=1.5, label='Corot 1-elem')
ax.plot(pd_dlat_cor10_2d, pd_P_cor10_2d, 'g-', lw=2, label='Corot 10-elem')
ax.axhline(P_cr_euler_Iz, color='grey', ls='--', alpha=0.5,
           label=f'Euler $P_{{cr}}$={P_cr_euler_Iz:.2f}')
ax.axhline(P_cr_pdelta_1_Iz, color='salmon', ls=':', alpha=0.5,
           label=f'P-Delta(1) $P_{{cr}}$={P_cr_pdelta_1_Iz:.1f}')
ax.set_xlabel('Lateral tip displacement')
ax.set_ylabel('Axial reaction P')
ax.set_title('2D: Pushdown — P vs lateral disp (strong axis)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# ---- Row 3, Col 2: 3D Pushdown (weak axis) ----
# Same as 2D pushdown but weak-axis buckling.
# Expect: Corot 10-elem plateaus at Euler P_cr_Iy ≈ 12.34
ax = axes[2, 1]
ax.plot(pd_dlat_pd1_3d, pd_P_pd1_3d, 'r--', lw=1.5, label='P-Delta 1-elem')
ax.plot(pd_dlat_cor1_3d, pd_P_cor1_3d, 'b:', lw=1.5, label='Corot 1-elem')
ax.plot(pd_dlat_cor10_3d, pd_P_cor10_3d, 'g-', lw=2, label='Corot 10-elem')
ax.axhline(P_cr_euler_Iy, color='grey', ls='--', alpha=0.5,
           label=f'Euler $P_{{cr}}$={P_cr_euler_Iy:.2f}')
ax.axhline(P_cr_pdelta_1_Iy, color='salmon', ls=':', alpha=0.5,
           label=f'P-Delta(1) $P_{{cr}}$={P_cr_pdelta_1_Iy:.1f}')
ax.set_xlabel('Lateral tip displacement Z')
ax.set_ylabel('Axial reaction P')
ax.set_title('3D: Pushdown — P vs lateral disp (weak axis)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout()
outpath = os.path.join(os.path.dirname(__file__), 'column_buckling_validation.png')
plt.savefig(outpath, dpi=150)
print(f"\nPlot saved to: {outpath}")
