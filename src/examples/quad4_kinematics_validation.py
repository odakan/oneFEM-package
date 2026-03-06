##-----------------------------------------------------------------------##
#  Quad4 Kinematics Validation — Publication Figures
#
#  Cantilever under dead-load end moment via horizontal force couple.
#  L=10, H=1, E=1.2e6, nu=0, plane stress, 20x2 mesh, 20 load steps.
#  Four formulations: Linear, TL, UL, Corot.
#
#  Force couple: +Fx at bottom tip node (tension), -Fx at top (compression),
#  where Fx = M/H. This produces upward bending. At small deformation,
#  the tip midline uy matches M*L^2/(2*EI) to within ~10% (shear
#  flexibility of the 20x2 Quad4 mesh).
#
#  Analytical reference: Dead-load elastica for horizontal force couple.
#  The couple moment at any cross-section = Fx * H * cos(theta_tip),
#  where theta_tip is the tip rotation. Curvature is uniform:
#    kappa = M / EI = Fx*H*cos(theta_tip) / EI
#  so theta(s) = kappa*s, and theta_tip satisfies the transcendental eq:
#    theta_tip = alpha * cos(theta_tip),  alpha = M0*L/(EI)
#
#  Key results:
#    - Linear diverges from nonlinear formulations past ~20% load
#    - TL == UL to machine precision (confirms UL fix)
#    - Corot tracks TL/UL closely
#
#  Figures (saved to docs/validation/quad4/):
#    1. quad4_val_deformed.png    — 4x4 grid deformed meshes
#    2. quad4_val_loaddispl.png   — load-displacement curves
#    3. quad4_val_convergence.png — Newton residual histories
#    4. quad4_val_summary.png     — error bar chart
##-----------------------------------------------------------------------##

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time as _time
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

from oneFEM.model import Domain
from oneFEM.model.node import Node22
from oneFEM.model.element.continuum.quad4 import Quad4
from oneFEM.model.material.nD.elastic_isotropic import ElasticIsotropic
from oneFEM.model.element.kinematics.continuum.linear import LinearContinuumKinematics
from oneFEM.model.element.kinematics.continuum.total_lagrangian import TotalLagrangianContinuumKinematics
from oneFEM.model.element.kinematics.continuum.updated_lagrangian import UpdatedLagrangianContinuumKinematics
from oneFEM.model.element.kinematics.continuum.corot import CorotContinuumKinematics
from oneFEM.model.pattern import Plain as PlainPattern
from oneFEM.model.tseries import Linear as LinearTS
from oneFEM.analysis import Analysis
from oneFEM.analysis.algorithm.newton_raphson import Newton
from oneFEM.analysis.constraints import Plain as PlainConstraints
from oneFEM.analysis.numberer import Plain as PlainNumberer
from oneFEM.analysis.system import FullGeneral
from oneFEM.analysis.integrator import LoadControl
from oneFEM.analysis.test import NormUnbalance

# ================================================================
# Parameters
# ================================================================

L = 10.0
H = 1.0
NX = 20
NY = 2
E_val = 1.2e6
nu_val = 0.0
t_val = 1.0
N_STEPS = 20

I_val = t_val * H**3 / 12.0
EI = E_val * I_val  # 1e5

# Moment for full 2*pi rollup (follower case); dead-load couple saturates well before
M_full = 2.0 * np.pi * EI / L  # 62831.85
F_x = M_full / H               # horizontal force at tip nodes

CHECK_STEPS = [5, 10, 15, 20]

FORMULATIONS = [
    ('Linear', None),
    ('TL', TotalLagrangianContinuumKinematics),
    ('UL', UpdatedLagrangianContinuumKinematics),
    ('Corot', CorotContinuumKinematics),
]

COLORS = {
    'Linear': '#4a90d9',
    'TL':     '#e8832a',
    'UL':     '#2ecc71',
    'Corot':  '#9b59b6',
}

BG_COLOR = '#0f0f14'
GRID_COLOR = '#1a1a24'
TEXT_COLOR = '#e0e0e0'
ANALYTICAL_COLOR = '#f1c40f'

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'validation', 'quad4')
os.makedirs(SAVE_DIR, exist_ok=True)


# ================================================================
# Dead-Load Elastica Analytical Solution
# ================================================================

def elastica_theta_tip(alpha):
    """Solve theta = alpha * cos(theta) for the tip rotation.

    alpha = M0*L/(EI) where M0 is the nominal couple moment (Fx*H).
    For dead horizontal couple, effective moment = M0*cos(theta_tip).
    Returns theta_tip in radians.
    """
    if abs(alpha) < 1e-14:
        return 0.0
    # theta - alpha*cos(theta) = 0; theta in (0, pi/2) for alpha > 0
    # For alpha > pi/2, the equation has no solution in (0,pi/2);
    # the maximum alpha for which a solution exists is at d/dtheta = 0:
    #   1 + alpha*sin(theta) = 0 (never zero for alpha > 0, theta in [0,pi/2])
    # Actually f(theta) = theta - alpha*cos(theta) is monotonically increasing,
    # f(0) = -alpha < 0, f(pi/2) = pi/2 > 0, so root always exists in (0, pi/2).
    return brentq(lambda th: th - alpha * np.cos(th), 0.0, np.pi / 2.0)


def elastica_tip(load_factor):
    """Tip displacement (ux, uy) from dead-load elastica at given load factor.

    load_factor: fraction of M_full applied (0 to 1).
    """
    alpha = load_factor * M_full * L / EI  # = load_factor * 2*pi
    theta_tip = elastica_theta_tip(alpha)
    if abs(theta_tip) < 1e-14:
        return 0.0, 0.0
    kappa = theta_tip / L
    # x(L) = sin(theta_tip) / kappa - L (displacement from original)
    ux = np.sin(theta_tip) / kappa - L
    uy = (1.0 - np.cos(theta_tip)) / kappa
    return ux, uy


def elastica_centerline(load_factor, n_pts=200):
    """Deformed centerline from dead-load elastica."""
    alpha = load_factor * M_full * L / EI
    theta_tip = elastica_theta_tip(alpha)
    if abs(theta_tip) < 1e-14:
        s = np.linspace(0, L, n_pts)
        return s, np.zeros(n_pts)
    kappa = theta_tip / L
    s = np.linspace(0, L, n_pts)
    xs = np.sin(kappa * s) / kappa
    ys = (1.0 - np.cos(kappa * s)) / kappa
    return xs, ys


# ================================================================
# Mesh Generation
# ================================================================

def rect_mesh(nx, ny, Lx, Ly):
    node_coords = {}
    node_grid = {}
    nid = 1
    dx, dy = Lx / nx, Ly / ny
    for j in range(ny + 1):
        for i in range(nx + 1):
            node_coords[nid] = (i * dx, j * dy)
            node_grid[(i, j)] = nid
            nid += 1
    elem_conn = {}
    eid = 1
    for j in range(ny):
        for i in range(nx):
            n1 = node_grid[(i, j)]
            n2 = node_grid[(i + 1, j)]
            n3 = node_grid[(i + 1, j + 1)]
            n4 = node_grid[(i, j + 1)]
            elem_conn[eid] = [n1, n2, n3, n4]
            eid += 1
    return node_coords, elem_conn, node_grid


# ================================================================
# Instrumented NormUnbalance
# ================================================================

class InstrumentedNormUnbalance(NormUnbalance):
    def __init__(self, tID=-1, tol=1e-6, maxIter=10, printFlag=0):
        super().__init__(tID, tol, maxIter, printFlag)
        self.all_steps = []
        self._current_norms = []

    def start(self):
        super().start()
        self._current_norms = []

    def test(self, norm_value):
        self._current_norms.append(norm_value)
        result = super().test(norm_value)
        if result == 0 or result == -2:
            self.all_steps.append(list(self._current_norms))
        return result


# ================================================================
# Build & Run One Formulation
# ================================================================

def run_formulation(name, kin_class, nSteps=N_STEPS):
    node_coords, elem_conn, node_grid = rect_mesh(NX, NY, L, H)

    model = Domain(nD=2)
    nodes = {}
    for nid, (x, y) in node_coords.items():
        nd = Node22(nid, coord=[x, y])
        nodes[nid] = nd
        model.add(nd)

    for j in range(NY + 1):
        nodes[node_grid[(0, j)]].setFix([True, True])

    mat = ElasticIsotropic(1, E_val, nu_val, type='PlaneStress')
    for eid, conn in elem_conn.items():
        kin = kin_class() if kin_class is not None else None
        elem = Quad4(eid, [nodes[c] for c in conn], mat,
                     kinematics=kin, thickness=t_val)
        model.add(elem)

    # Horizontal force couple: +Fx bottom (tension), -Fx top (compression)
    tip_bot = node_grid[(NX, 0)]
    tip_top = node_grid[(NX, NY)]
    tip_mid = node_grid[(NX, NY // 2)]

    ts = LinearTS(1, factor=1.0)
    pat = PlainPattern(1, tseries=ts,
                       load=[[tip_bot,  F_x, 0.0],
                             [tip_top, -F_x, 0.0]])
    model.add(pat)

    dt_val = 1.0 / nSteps
    alg = Newton(1, tangent='current')
    ctest = InstrumentedNormUnbalance(1, tol=1e-8, maxIter=50)
    analysis = Analysis(1, algorithm=alg, constraints=PlainConstraints(1),
                        integrator=LoadControl(1), system=FullGeneral(1),
                        test=ctest)

    tip_x_hist = [float(node_coords[tip_mid][0])]
    tip_y_hist = [float(node_coords[tip_mid][1])]
    snapshots = {}

    converged = True
    t0 = _time.perf_counter()

    model._domain()
    analysis._organize(model)
    integrator_obj = analysis._solution_integrator

    for step in range(nSteps):
        assembly_time = analysis._time + dt_val
        model._assemble(time=assembly_time)
        integrator_obj.newStep(model, dt_val, analysis._time)
        try:
            analysis._solution_algorithm.solve(
                model, analysis.uu, analysis.pp,
                integrator_obj, analysis._system_of_equations, ctest)
        except Exception as e:
            print("    [{}] Step {} failed: {}".format(name, step + 1, e))
            converged = False
            break
        integrator_obj.commit(model)
        model._record(time=analysis._time + dt_val)
        analysis._time += dt_val

        u = nodes[tip_mid]._getCommitDisp()
        tip_x_hist.append(node_coords[tip_mid][0] + float(u[0]))
        tip_y_hist.append(node_coords[tip_mid][1] + float(u[1]))

        if step == 0:
            ux_1 = tip_x_hist[-1] - tip_x_hist[0]
            uy_1 = tip_y_hist[-1] - tip_y_hist[0]
            lf_1 = dt_val
            ux_e, uy_e = elastica_tip(lf_1)
            print("    Step 1: ux={:.4e} (ela={:.4e}), uy={:.4e} (ela={:.4e})".format(
                ux_1, ux_e, uy_1, uy_e))

        if (step + 1) in CHECK_STEPS:
            snap = {}
            for nid_s, (x0, y0) in node_coords.items():
                uc = nodes[nid_s]._getCommitDisp()
                snap[nid_s] = (x0 + float(uc[0]), y0 + float(uc[1]))
            snapshots[step + 1] = snap

    wall_time = _time.perf_counter() - t0

    return {
        'name': name,
        'converged': converged,
        'tip_x': tip_x_hist,
        'tip_y': tip_y_hist,
        'snapshots': snapshots,
        'residuals': ctest.all_steps,
        'wall_time': wall_time,
        'node_coords': node_coords,
        'elem_conn': elem_conn,
    }


# ================================================================
# Dark-Theme Style
# ================================================================

def style_dark(fig, axes):
    fig.patch.set_facecolor(BG_COLOR)
    for ax in np.array(axes).flat:
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.5)


# ================================================================
# Figure 1: Deformed Meshes (4 rows x 4 cols)
# ================================================================

def plot_deformed(all_results):
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    style_dark(fig, axes)

    for col, res in enumerate(all_results):
        name = res['name']
        color = COLORS[name]
        node_coords = res['node_coords']
        elem_conn = res['elem_conn']

        for row, step_num in enumerate(CHECK_STEPS):
            ax = axes[row, col]
            ax.grid(False)

            # Undeformed mesh (faint)
            for eid, conn in elem_conn.items():
                verts = [node_coords[n] for n in conn] + [node_coords[conn[0]]]
                xs, ys = zip(*verts)
                ax.plot(xs, ys, color='#333340', linewidth=0.4, zorder=1)

            # Deformed mesh
            if step_num in res['snapshots']:
                snap = res['snapshots'][step_num]
                polys = []
                for eid, conn in elem_conn.items():
                    verts = [snap[n] for n in conn]
                    polys.append(mpatches.Polygon(verts, closed=True))
                pc = PatchCollection(polys, alpha=0.7, facecolor=color,
                                     edgecolor=color, linewidth=0.3)
                ax.add_collection(pc)

            # Elastica centerline
            lf = step_num / N_STEPS
            cx, cy = elastica_centerline(lf)
            ax.plot(cx, cy, '--', color=ANALYTICAL_COLOR, linewidth=1.5, zorder=10)

            ax.set_aspect('equal')
            pad = L * 0.15
            ax.set_xlim(-pad, L + pad)
            ax.set_ylim(-pad, L * 0.6 + pad)
            ax.tick_params(labelsize=7)

            if row == 0:
                ax.set_title(name, fontsize=13, fontweight='bold', color=color)
            if col == 0:
                ax.set_ylabel('Step {} (LF={:.0f}%)'.format(step_num, lf * 100),
                              fontsize=10, color=TEXT_COLOR)
            if row < 3:
                ax.set_xticklabels([])

    fig.suptitle('Dead-Load End Moment -- Deformed Meshes\n'
                 '(yellow dashed = dead-load elastica reference)',
                 fontsize=14, color=TEXT_COLOR, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(SAVE_DIR, 'quad4_val_deformed.png')
    fig.savefig(path, dpi=200, facecolor=BG_COLOR)
    plt.close(fig)
    print("  Saved: {}".format(path))


# ================================================================
# Figure 2: Load-Displacement Curves
# ================================================================

def plot_load_disp(all_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    style_dark(fig, [ax1, ax2])

    # Elastica curve
    n_pts = 200
    lfs = np.linspace(0, 1, n_pts)
    ux_e = np.zeros(n_pts)
    uy_e = np.zeros(n_pts)
    for i, lf in enumerate(lfs):
        ux_e[i], uy_e[i] = elastica_tip(lf)

    ax1.plot(lfs, ux_e, '--', color=ANALYTICAL_COLOR, linewidth=2,
             label='Elastica', zorder=10)
    ax2.plot(lfs, uy_e, '--', color=ANALYTICAL_COLOR, linewidth=2,
             label='Elastica', zorder=10)

    for res in all_results:
        name = res['name']
        color = COLORS[name]
        nS = len(res['tip_x']) - 1
        lf_arr = np.linspace(0, 1, nS + 1)
        ux = np.array(res['tip_x']) - res['tip_x'][0]
        uy = np.array(res['tip_y']) - res['tip_y'][0]

        ax1.plot(lf_arr, ux, '-', color=color, linewidth=1.5, label=name, alpha=0.9)
        ax2.plot(lf_arr, uy, '-', color=color, linewidth=1.5, label=name, alpha=0.9)

        for step in CHECK_STEPS:
            if step <= nS:
                ax1.plot(lf_arr[step], ux[step], 'o', color=color, markersize=6,
                         markeredgecolor='white', markeredgewidth=0.5, zorder=5)
                ax2.plot(lf_arr[step], uy[step], 'o', color=color, markersize=6,
                         markeredgecolor='white', markeredgewidth=0.5, zorder=5)

    for ax, comp in [(ax1, r'$u_x$'), (ax2, r'$u_y$')]:
        ax.set_xlabel('Load Factor')
        ax.set_ylabel('{} (tip midline)'.format(comp))
        ax.set_title('Tip {} vs Load Factor'.format(comp), color=TEXT_COLOR)
        ax.legend(facecolor='#1a1a24', edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR, fontsize=9)

    fig.suptitle('Dead-Load End Moment -- Elastica Reference\n'
                 '(horizontal force couple on 20x2 Quad4 mesh)',
                 fontsize=13, color=TEXT_COLOR, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, 'quad4_val_loaddispl.png')
    fig.savefig(path, dpi=200, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: {}".format(path))


# ================================================================
# Figure 3: Newton Convergence (2x2)
# ================================================================

def plot_convergence(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    style_dark(fig, axes)

    for idx, res in enumerate(all_results):
        ax = axes.flat[idx]
        name = res['name']
        color = COLORS[name]

        for step_idx, norms in enumerate(res['residuals']):
            step_num = step_idx + 1
            iters = np.arange(1, len(norms) + 1)
            if step_num in CHECK_STEPS:
                ax.semilogy(iters, norms, '-o', color=color, linewidth=1.5,
                            markersize=4, alpha=0.9,
                            label='Step {}'.format(step_num), zorder=5)
            else:
                ax.semilogy(iters, norms, '-', color='#555555', linewidth=0.5,
                            alpha=0.4, zorder=1)

        ax.axhline(y=1e-8, color=ANALYTICAL_COLOR, linestyle=':', linewidth=1.0,
                   label='tol = 1e-8', zorder=8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Unbalanced Norm')
        ax.set_title('{} -- Newton Residual History'.format(name),
                     fontsize=11, color=color, fontweight='bold')
        ax.legend(facecolor='#1a1a24', edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR, fontsize=8, loc='upper right')
        ax.set_xlim(0.5, None)

    fig.suptitle('Dead-Load End Moment -- Newton Convergence',
                 fontsize=14, color=TEXT_COLOR, fontweight='bold', y=1.0)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, 'quad4_val_convergence.png')
    fig.savefig(path, dpi=200, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: {}".format(path))


# ================================================================
# Figure 4: Error Summary (Grouped Bar Chart)
# ================================================================

def plot_summary(all_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    style_dark(fig, [ax1, ax2])

    n_form = len(all_results)
    n_check = len(CHECK_STEPS)
    bar_width = 0.8 / n_form
    x_base = np.arange(n_check)

    for ax, component, label in [
        (ax1, 'ux', r'$u_x$ Error vs Elastica (%)'),
        (ax2, 'uy', r'$u_y$ Error vs Elastica (%)')
    ]:
        for fi, res in enumerate(all_results):
            name = res['name']
            color = COLORS[name]
            errors = []
            for ci, step in enumerate(CHECK_STEPS):
                lf = step / N_STEPS
                ux_ana, uy_ana = elastica_tip(lf)
                nS = len(res['tip_x']) - 1
                if step <= nS:
                    ux_num = res['tip_x'][step] - res['tip_x'][0]
                    uy_num = res['tip_y'][step] - res['tip_y'][0]
                else:
                    ux_num, uy_num = 0.0, 0.0

                if component == 'ux':
                    ref = abs(ux_ana) if abs(ux_ana) > 1e-10 else 1.0
                    err = abs(ux_num - ux_ana) / ref * 100.0
                else:
                    ref = abs(uy_ana) if abs(uy_ana) > 1e-10 else 1.0
                    err = abs(uy_num - uy_ana) / ref * 100.0
                errors.append(min(err, 200.0))

            x_pos = x_base + fi * bar_width
            bars = ax.bar(x_pos, errors, bar_width * 0.9, color=color,
                          alpha=0.85, label=name, edgecolor='white', linewidth=0.3)
            for bar, err in zip(bars, errors):
                if 0.5 < err < 200:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.5,
                            '{:.1f}'.format(err), ha='center', va='bottom',
                            fontsize=7, color=TEXT_COLOR)

        ax.axhline(y=5.0, color=ANALYTICAL_COLOR, linestyle='--', linewidth=1.0,
                   label='5% threshold', zorder=8)
        ax.set_xticks(x_base + bar_width * (n_form - 1) / 2)
        ax.set_xticklabels(['LF={:.0f}%'.format(s / N_STEPS * 100)
                            for s in CHECK_STEPS], fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=12, color=TEXT_COLOR)
        ax.legend(facecolor='#1a1a24', edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR, fontsize=8, loc='upper left')

    fig.suptitle('Dead-Load End Moment -- Error vs Elastica Reference',
                 fontsize=13, color=TEXT_COLOR, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, 'quad4_val_summary.png')
    fig.savefig(path, dpi=200, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: {}".format(path))


# ================================================================
# Main
# ================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Quad4 Kinematics Validation — Dead-Load End Moment (Force Couple)")
    print("  L={}, H={}, E={:.1e}, nu={}, t={}, {}x{} mesh, {} steps".format(
        L, H, E_val, nu_val, t_val, NX, NY, N_STEPS))
    print("  EI = {:.4e},  M_full = {:.4e} (2*pi*EI/L)".format(EI, M_full))
    print("  Force couple: Fx = M/H = {:.4e} at tip".format(F_x))
    print("  Reference: dead-load elastica (theta = alpha*cos(theta))")
    print("=" * 70)

    # Print elastica reference values
    print("\n  Elastica reference (dead-load couple):")
    for step in CHECK_STEPS:
        lf = step / N_STEPS
        ux_e, uy_e = elastica_tip(lf)
        alpha = lf * M_full * L / EI
        theta = elastica_theta_tip(alpha)
        print("    LF={:.0f}%: theta={:.4f} rad ({:.1f} deg), ux={:.4f}, uy={:.4f}".format(
            lf * 100, theta, np.degrees(theta), ux_e, uy_e))

    all_results = []
    for name, kin_class in FORMULATIONS:
        print("\n  Running {}...".format(name))
        res = run_formulation(name, kin_class)
        all_results.append(res)
        status = "CONVERGED" if res['converged'] else "FAILED"
        ux_final = res['tip_x'][-1] - res['tip_x'][0]
        uy_final = res['tip_y'][-1] - res['tip_y'][0]
        print("    {} -- {} in {:.2f}s  (ux={:.6f}, uy={:.6f})".format(
            name, status, res['wall_time'], ux_final, uy_final))

    # Error table
    print("\n" + "-" * 70)
    print("  Error vs dead-load elastica:")
    header = "  {:>10s}".format('LF')
    for res in all_results:
        header += "  {:>10s}".format(res['name'])
    print(header)
    print("  " + "-" * (12 + 12 * len(all_results)))

    for step in CHECK_STEPS:
        lf = step / N_STEPS
        ux_ana, uy_ana = elastica_tip(lf)
        for comp, val_ana, label in [('ux', ux_ana, 'ux'), ('uy', uy_ana, 'uy')]:
            row = "  {} {:>4.0f}%".format(label, lf * 100)
            for res in all_results:
                nS = len(res['tip_x']) - 1
                if step <= nS:
                    val_num = (res['tip_x'][step] - res['tip_x'][0] if comp == 'ux'
                               else res['tip_y'][step] - res['tip_y'][0])
                else:
                    val_num = 0.0
                ref = abs(val_ana) if abs(val_ana) > 1e-10 else 1.0
                err = abs(val_num - val_ana) / ref * 100.0
                row += "  {:>9.2f}%".format(err)
            print(row)

    # TL vs UL agreement
    print("\n  TL vs UL agreement (confirms UL fix):")
    tl_res = next(r for r in all_results if r['name'] == 'TL')
    ul_res = next(r for r in all_results if r['name'] == 'UL')
    all_pass = True
    for step in CHECK_STEPS:
        nS = min(len(tl_res['tip_x']) - 1, len(ul_res['tip_x']) - 1)
        if step <= nS:
            ux_tl = tl_res['tip_x'][step] - tl_res['tip_x'][0]
            uy_tl = tl_res['tip_y'][step] - tl_res['tip_y'][0]
            ux_ul = ul_res['tip_x'][step] - ul_res['tip_x'][0]
            uy_ul = ul_res['tip_y'][step] - ul_res['tip_y'][0]
            err = max(abs(ux_tl - ux_ul), abs(uy_tl - uy_ul))
            status = "PASS" if err < 1e-6 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print("    Step {:2d}: max|TL-UL| = {:.2e}  {}".format(
                step, err, status))

    # Wall times
    print("\n  Wall times:")
    for res in all_results:
        print("    {:>8s}: {:.2f}s".format(res['name'], res['wall_time']))

    # Generate figures
    print("\n  Generating figures...")
    plot_deformed(all_results)
    plot_load_disp(all_results)
    plot_convergence(all_results)
    plot_summary(all_results)

    print("\n" + "=" * 70)
    if all_pass:
        print("  TL == UL: ALL PASS (UL fix validated)")
    else:
        print("  TL == UL: SOME FAILED")
    print("  Done.")
    print("=" * 70)
