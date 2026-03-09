##-----------------------------------------------------------------------##
#  3D Geometric Nonlinearity Comparison Study:
#  Bathe & Bolourchi (1979) Curved Cantilever — Hex8
#
#  45-degree arch in the X-Y plane, R=100, cross-section 1x1,
#  E=1e7, nu=0.0. Fixed end near Y-axis (theta=90deg),
#  free tip at theta=45deg.
#  Tip load P=600 in the +Z direction (out of arch plane).
#
#  Mesh: 12x1x1 Hex8 (12 along arc, 1 radial, 1 width).
#  Analysis: Newton + LoadControl, 20 equal load steps (default).
#  Formulations: Linear, TL, UL, Corot.
#
#  Bathe & Bolourchi 1979 reference at P=600 (8 cubic beam elements):
#    |ux| = 13.4,  |uy| = 23.5,  |uz| = 53.4  (magnitudes from paper)
#  Sign convention (arch in X-Y plane, load in +Z):
#    ux = -13.4  (inward toward center in X)
#    uy = +23.5  (tangential, positive Y at theta=45deg)
#    uz = +53.4  (in load direction)
#
#  Sign fix (2026-03-07): uy reference changed from -23.5 to +23.5.
#  The paper reports magnitudes only. The negative sign was incorrectly
#  assumed (radial inward). At theta=45deg, the dominant in-plane motion
#  is tangential toward the fixed end, whose Y-component is positive.
#
#  ----------------------------------------------------------------
#  Mesh refinement study (TL formulation, all meshes converged)
#  ----------------------------------------------------------------
#
#    Mesh      | Incomp | ux_err% | uy_err% | uz_err%
#    ----------+--------+---------+---------+--------
#    12x1x1    |   No   |  95.4%  |  91.8%  |  75.6%
#    24x1x1    |   No   |  74.5%  |  87.0%  |  56.9%
#    48x1x1    |   No   |  16.4%  |  70.9%  |  27.3%
#    24x4x4    |   No   |  73.4%  |  85.4%  |  55.6%
#    48x4x4    |   No   |   —     |   —     |  26.6%
#    12x1x1    |  Yes   |  39.7%  | 107%    |  51.2%
#    24x1x1    |  Yes   |   5.2%  |  77.1%  |  23.1%
#    48x4x4    |  Yes   |  21.9%  |  62.5%  |  12.7%
#    Bathe ref |   —    |   0%    |   0%    |   0%
#
#  ----------------------------------------------------------------
#  Conclusions
#  ----------------------------------------------------------------
#
#  1. Incompatible modes halve uz error at equivalent mesh density.
#     48x4x4 standard: 26.6% uz error -> 48x4x4 incompatible: 12.7%.
#     24x1x1 standard: 56.9% -> 24x1x1 incompatible: 23.1%.
#     The Wilson modes effectively reduce shear locking in the load
#     direction, consistent with their design for bending-dominated
#     response.
#
#  2. uy remains locking-dominated on curved geometry. Even at 48x4x4
#     incompatible the uy error is 62.5%. The tangential displacement
#     requires coupled membrane-bending response that Wilson modes
#     (designed for flat-element shear locking) cannot unlock on curved
#     elements. The G matrix built from J0 at the element center assumes
#     locally flat geometry; curvature coupling defeats this assumption.
#
#  3. EAS formulation (Simo & Rifai 1990) is required for a full
#     membrane locking cure on curved geometry. Wilson incompatible
#     modes are a partial remedy (effective for uz/shear locking) but
#     insufficient for the membrane component (uy). Arc-direction
#     refinement is more effective than cross-section refinement when
#     using Wilson modes on curved elements.
#
#  4. Corot (EICR) diverges at ~70-75% load on this problem regardless
#     of step count (tested 20/40/60/80 steps). This is a formulation
#     limit for combined bending + torsion + large rotation, not a
#     Newton step-size issue.
#
#  Figures saved to docs/validation/hex8/:
#    1. hex8_curved_deformed.png      — 4x4 grid deformed meshes
#    2. hex8_curved_loaddispl.png     — load-displacement curves
#    3. hex8_curved_convergence.png   — Newton residual histories
#    4. hex8_curved_error.png         — error bar chart vs Bathe ref
##-----------------------------------------------------------------------##

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time as _time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from oneFEM.model import Domain
from oneFEM.model.node import Node33
from oneFEM.model.element.continuum.hex8 import Hex8
from oneFEM.model.kinematics.continuum.cauchy.linear import LinearContinuumKinematics
from oneFEM.model.kinematics.continuum.cauchy.total_lagrangian import TotalLagrangianContinuumKinematics
from oneFEM.model.kinematics.continuum.cauchy.updated_lagrangian import UpdatedLagrangianContinuumKinematics
from oneFEM.model.kinematics.continuum.cauchy.corot import CorotContinuumKinematics
from oneFEM.model.material.nD.elastic_isotropic import ElasticIsotropic
from oneFEM.model.pattern import Plain as PlainPattern
from oneFEM.model.tseries import Linear as LinearTS
from oneFEM.analysis import Analysis
from oneFEM.analysis.algorithm.newton_raphson import Newton
from oneFEM.analysis.constraints import Plain as PlainConstraints
from oneFEM.analysis.numberer import Plain as PlainNumberer
from oneFEM.analysis.system import FullGeneral, UmfPackSOE
from oneFEM.analysis.numberer.rcm import RCM
from oneFEM.analysis.integrator import LoadControl
from oneFEM.analysis.test import NormUnbalance


# ================================================================
# Parameters
# ================================================================

R = 100.0         # arch radius
H = 1.0           # cross-section height (radial)
W = 1.0           # cross-section width (Y)
E_VAL = 1e7
NU_VAL = 0.0
P_TOTAL = 600.0   # tip load in +Z (out of arch plane)
N_ARC = 12        # elements along arc
N_STEPS = 20

# Bathe & Bolourchi 1979 reference at full load (P=600, +Z)
BATHE_REF = {'ux': -13.4, 'uy': 23.5, 'uz': 53.4}

CHECK_STEPS = [5, 10, 15, 20]  # 25%, 50%, 75%, 100% load

FORMULATIONS = [
    ('Linear', LinearContinuumKinematics),
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

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'validation', 'hex8')
os.makedirs(SAVE_DIR, exist_ok=True)


# ================================================================
# Curved Mesh Builder
# ================================================================

def build_curved_mesh(n_arc=N_ARC, n_rad=1, n_width=1):
    """Build 45-degree arch Hex8 mesh in X-Y plane.

    Arc stations (theta measured from X-axis):
      iz=0      -> theta=90deg (near Y-axis) -> FIXED end
      iz=n_arc  -> theta=45deg               -> FREE end (tip load in +Z)
    Cross-section: HxW, n_rad elements radially (ix) x n_width elements in Z (iw).

    :param n_arc: number of elements along the arc
    :param n_rad: number of elements in radial direction (cross-section height)
    :param n_width: number of elements in width/Z direction
    """
    nr = n_rad + 1      # nodes in radial
    nw = n_width + 1    # nodes in width
    nps = nw * nr       # nodes per arc station

    node_coords = {}
    for iz in range(n_arc + 1):
        theta = np.pi / 2.0 - iz * np.pi / (4.0 * n_arc)
        for iw in range(nw):
            for ix in range(nr):
                nid = iz * nps + iw * nr + ix + 1
                r = R - H / 2.0 + ix * H / n_rad
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z = -W / 2.0 + iw * W / n_width
                node_coords[nid] = (x, y, z)

    elem_conn = {}
    eid = 1
    for iz in range(n_arc):
        for iw in range(n_width):
            for ix in range(n_rad):
                def nid_f(jz, jw, jx):
                    return jz * nps + jw * nr + jx + 1
                # Hex8: ξ₁→arc, ξ₂→radial, ξ₃→width(Z)
                elem_conn[eid] = [
                    nid_f(iz,   iw,   ix),   nid_f(iz+1, iw,   ix),
                    nid_f(iz+1, iw,   ix+1), nid_f(iz,   iw,   ix+1),
                    nid_f(iz,   iw+1, ix),   nid_f(iz+1, iw+1, ix),
                    nid_f(iz+1, iw+1, ix+1), nid_f(iz,   iw+1, ix+1),
                ]
                eid += 1

    # Fixed-end node IDs (iz=0)
    fixed_nids = [0 * nps + iw * nr + ix + 1
                  for iw in range(nw) for ix in range(nr)]
    # Free-end node IDs (iz=n_arc)
    free_nids = [n_arc * nps + iw * nr + ix + 1
                 for iw in range(nw) for ix in range(nr)]

    return node_coords, elem_conn, free_nids, fixed_nids


# ================================================================
# Hex8 edge/face helpers for plotting
# ================================================================

HEX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]

HEX_FACES = [
    [0, 1, 2, 3], [4, 5, 6, 7],  # bottom, top
    [0, 1, 5, 4], [1, 2, 6, 5],  # front, right
    [2, 3, 7, 6], [3, 0, 4, 7],  # back, left
]


# ================================================================
# Instrumented convergence test
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

def run_formulation(name, kin_class, nSteps=N_STEPS,
                    n_arc=N_ARC, n_rad=1, n_width=1,
                    system=None, numberer=None):
    node_coords, elem_conn, free_nids, fixed_nids = build_curved_mesh(
        n_arc, n_rad, n_width)

    model = Domain()
    nodes = {}
    for nid in sorted(node_coords.keys()):
        x, y, z = node_coords[nid]
        nd = Node33(nid, coord=[x, y, z])
        nodes[nid] = nd
        model.add(nd)

    # Fix clamped end
    for nid in fixed_nids:
        nodes[nid].setFix([True, True, True])

    mat = ElasticIsotropic(1, E_VAL, NU_VAL, type='3D')

    for eid in sorted(elem_conn.keys()):
        conn = elem_conn[eid]
        # Per-element kinematics instance (critical for stateful formulations)
        if name == 'Linear':
            kin = LinearContinuumKinematics(bbar=False)
        else:
            kin = kin_class()
        elem = Hex8(eid, [nodes[c] for c in conn], mat, kinematics=kin,
                    incompatible=False)
        model.add(elem)

    # Tip load: P in +Z distributed on 4 free-end nodes
    f_per_node = P_TOTAL / len(free_nids)
    ts = LinearTS(1, factor=1.0)
    load_list = [[nid, 0.0, 0.0, f_per_node] for nid in free_nids]
    pat = PlainPattern(1, tseries=ts, load=load_list)
    model.add(pat)

    dt_val = 1.0 / nSteps
    alg = Newton(1, tangent='current')
    # NormDispIncr-equivalent: tol scaled by stiffness O(1e4) from E=1e7, R=100.
    # tol_disp ~ 1e-9 -> tol_force = tol_disp * K ~ 1e-9 * 1e4 = 1e-5.
    ctest = InstrumentedNormUnbalance(1, tol=1e-5, maxIter=50)
    if system is None:
        system = FullGeneral(1)
    if numberer is None:
        numberer = PlainNumberer(1)
    analysis = Analysis(1, algorithm=alg, constraints=PlainConstraints(1),
                        integrator=LoadControl(1), system=system,
                        numberer=numberer, test=ctest)

    # Tip displacement history (average of free-end nodes)
    tip_ux = [0.0]
    tip_uy = [0.0]
    tip_uz = [0.0]
    snapshots = {}

    converged = True
    t0 = _time.perf_counter()

    analysis._analyze(model, nSteps=0, dt=0.0)

    # Corot diagnostic: check strain at GP 0 of element 1 after _analyze()
    if name == 'Corot':
        elem1 = model.elements[0]
        strain0 = elem1._kinematics.getStrain(0)
        if strain0 is not None:
            s_vec = strain0.make_vector()
            s_norm = np.linalg.norm(s_vec)
            print("    [Corot diagnostic] Strain at GP 0 after _analyze(): "
                  "norm = {:.2e}".format(s_norm))
            if s_norm > 1e-12:
                print("    [Corot diagnostic] WARNING: nonzero initial strain!")
                print("    [Corot diagnostic] strain = {}".format(s_vec))
                R_init = elem1._kinematics._R
                print("    [Corot diagnostic] R = \n{}".format(R_init))
        else:
            print("    [Corot diagnostic] Strain at GP 0 is None after _analyze()")

    integrator_obj = analysis._solution_integrator

    for step in range(nSteps):
        assembly_time = analysis._time + dt_val
        analysis._assembleF(model, time=assembly_time)
        integrator_obj.newStep(model, dt_val, analysis._time)
        try:
            analysis._solution_algorithm.solve(
                model, analysis.uu, analysis.pp,
                integrator_obj, analysis._assembly_system, ctest)
        except Exception as e:
            print("    [{}] Step {} failed: {}".format(name, step + 1, e))
            converged = False
            break
        integrator_obj.commit(model)
        model._record(time=analysis._time + dt_val)
        analysis._time += dt_val

        # Record tip displacement
        ux_vals, uy_vals, uz_vals = [], [], []
        for nid in free_nids:
            u = nodes[nid]._getCommitDisp()
            ux_vals.append(float(u[0]))
            uy_vals.append(float(u[1]))
            uz_vals.append(float(u[2]))
        tip_ux.append(np.mean(ux_vals))
        tip_uy.append(np.mean(uy_vals))
        tip_uz.append(np.mean(uz_vals))

        # Snapshot for deformed mesh
        if (step + 1) in CHECK_STEPS:
            snap = {}
            for nid_s, (x0, y0, z0) in node_coords.items():
                uc = nodes[nid_s]._getCommitDisp()
                snap[nid_s] = (x0 + float(uc[0]),
                               y0 + float(uc[1]),
                               z0 + float(uc[2]))
            snapshots[step + 1] = snap

    wall_time = _time.perf_counter() - t0

    return {
        'name': name,
        'converged': converged,
        'tip_ux': tip_ux,
        'tip_uy': tip_uy,
        'tip_uz': tip_uz,
        'snapshots': snapshots,
        'residuals': ctest.all_steps,
        'wall_time': wall_time,
        'node_coords': node_coords,
        'elem_conn': elem_conn,
    }


# ================================================================
# Dark-Theme Styling
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


def style_dark_3d(fig, axes):
    fig.patch.set_facecolor(BG_COLOR)
    for ax in np.array(axes).flat:
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.zaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(GRID_COLOR)
        ax.yaxis.pane.set_edgecolor(GRID_COLOR)
        ax.zaxis.pane.set_edgecolor(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.3, alpha=0.3)


def _draw_hex_wireframe(ax, elem_conn, pos, color, linewidth=0.5, alpha=1.0):
    """Draw wireframe edges for all Hex8 elements."""
    for eid, conn in elem_conn.items():
        pts = [pos[n] for n in conn]
        for i, j in HEX_EDGES:
            ax.plot3D([pts[i][0], pts[j][0]],
                      [pts[i][1], pts[j][1]],
                      [pts[i][2], pts[j][2]],
                      color=color, linewidth=linewidth, alpha=alpha)


def _draw_hex_filled(ax, elem_conn, pos, color, alpha=0.3):
    """Draw filled Hex8 faces."""
    polys = []
    for eid, conn in elem_conn.items():
        pts = [pos[n] for n in conn]
        for face in HEX_FACES:
            verts = [pts[f] for f in face]
            polys.append(verts)
    pc = Poly3DCollection(polys, alpha=alpha, facecolor=color,
                          edgecolor=color, linewidth=0.2)
    ax.add_collection3d(pc)


# ================================================================
# Figure 1: Deformed Meshes (4 rows x 4 cols, 3D)
# ================================================================

def plot_deformed(all_results):
    fig = plt.figure(figsize=(22, 18))
    style_dark(fig, [])  # style the figure only
    axes = []
    for idx in range(16):
        ax = fig.add_subplot(4, 4, idx + 1, projection='3d')
        axes.append(ax)
    style_dark_3d(fig, axes)

    for col, res in enumerate(all_results):
        name = res['name']
        color = COLORS[name]
        node_coords = res['node_coords']
        elem_conn = res['elem_conn']

        for row, step_num in enumerate(CHECK_STEPS):
            ax = axes[row * 4 + col]
            ax.grid(False)

            # Reference mesh (grey wireframe)
            _draw_hex_wireframe(ax, elem_conn, node_coords,
                                '#333340', linewidth=0.3, alpha=0.5)

            # Deformed mesh (filled + wireframe)
            if step_num in res['snapshots']:
                snap = res['snapshots'][step_num]
                _draw_hex_filled(ax, elem_conn, snap, color, alpha=0.35)
                _draw_hex_wireframe(ax, elem_conn, snap, color,
                                    linewidth=0.6, alpha=0.9)

            ax.view_init(elev=25, azim=-55)
            ax.set_xlim(-20, 110)
            ax.set_ylim(-70, 10)
            ax.set_zlim(-20, 110)
            ax.tick_params(labelsize=5, pad=0)
            ax.set_xlabel('X', fontsize=6, labelpad=1)
            ax.set_ylabel('Y', fontsize=6, labelpad=1)
            ax.set_zlabel('Z', fontsize=6, labelpad=1)

            if row == 0:
                ax.set_title(name, fontsize=12, fontweight='bold',
                             color=color, pad=2)

            lf = step_num / N_STEPS
            if col == 0:
                ax.text2D(0.02, 0.95, 'LF={:.0f}%'.format(lf * 100),
                          transform=ax.transAxes, fontsize=9,
                          color=TEXT_COLOR, verticalalignment='top')

    fig.suptitle('Bathe & Bolourchi Curved Cantilever -- Deformed Hex8 Meshes\n'
                 '(grey = reference, colored = deformed)',
                 fontsize=14, color=TEXT_COLOR, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(SAVE_DIR, 'hex8_curved_deformed.png')
    fig.savefig(path, dpi=180, facecolor=BG_COLOR)
    plt.close(fig)
    print("  Saved: {}".format(path))


# ================================================================
# Figure 2: Load-Displacement Curves (3 panels)
# ================================================================

def plot_load_disp(all_results):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    style_dark(fig, [ax1, ax2, ax3])

    for res in all_results:
        name = res['name']
        color = COLORS[name]
        nS = len(res['tip_ux']) - 1
        lf_arr = np.linspace(0, 1, nS + 1)

        ax1.plot(lf_arr, res['tip_ux'], '-', color=color, linewidth=1.5,
                 label=name, alpha=0.9)
        ax2.plot(lf_arr, res['tip_uy'], '-', color=color, linewidth=1.5,
                 label=name, alpha=0.9)
        ax3.plot(lf_arr, res['tip_uz'], '-', color=color, linewidth=1.5,
                 label=name, alpha=0.9)

    # Bathe reference markers at LF=1.0
    for ax, comp in [(ax1, 'ux'), (ax2, 'uy'), (ax3, 'uz')]:
        ax.plot(1.0, BATHE_REF[comp], '*', color=ANALYTICAL_COLOR,
                markersize=14, markeredgecolor='white', markeredgewidth=0.5,
                zorder=10, label='Bathe ref')

    for ax, comp in [(ax1, r'$u_x$'), (ax2, r'$u_y$'), (ax3, r'$u_z$')]:
        ax.set_xlabel('Load Factor', fontsize=10)
        ax.set_ylabel('{} (tip average)'.format(comp), fontsize=10)
        ax.set_title('Tip {}'.format(comp), color=TEXT_COLOR, fontsize=11)
        ax.legend(facecolor='#1a1a24', edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR, fontsize=8)

    fig.suptitle('Bathe & Bolourchi Curved Cantilever -- Tip Displacements\n'
                 'R={}, E={:.0e}, nu={}, P={}, 12x1x1 Hex8'.format(
                     R, E_VAL, NU_VAL, P_TOTAL),
                 fontsize=13, color=TEXT_COLOR, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, 'hex8_curved_loaddispl.png')
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

        ax.axhline(y=1e-4, color=ANALYTICAL_COLOR, linestyle=':', linewidth=1.0,
                   label='tol = 1e-4', zorder=8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Unbalanced Norm')
        ax.set_title('{} -- Newton Residual'.format(name),
                     fontsize=11, color=color, fontweight='bold')
        ax.legend(facecolor='#1a1a24', edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR, fontsize=8, loc='upper right')
        ax.set_xlim(0.5, None)

    fig.suptitle('Bathe & Bolourchi Curved Cantilever -- Newton Convergence',
                 fontsize=14, color=TEXT_COLOR, fontweight='bold', y=1.0)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, 'hex8_curved_convergence.png')
    fig.savefig(path, dpi=200, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: {}".format(path))


# ================================================================
# Figure 4: Error Bar Chart vs Bathe Reference
# ================================================================

def plot_error(all_results):
    """Grouped bar chart: % error vs Bathe at check steps.

    At 100% load, the Bathe reference is exact. At intermediate loads,
    it is linearly interpolated (approximate).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    style_dark(fig, axes)

    components = ['ux', 'uy', 'uz']
    comp_labels = [r'$u_x$', r'$u_y$', r'$u_z$']
    n_form = len(all_results)
    n_check = len(CHECK_STEPS)
    bar_width = 0.8 / n_form
    x_base = np.arange(n_check)

    for ci, (comp, label) in enumerate(zip(components, comp_labels)):
        ax = axes[ci]
        ref_full = BATHE_REF[comp]

        for fi, res in enumerate(all_results):
            name = res['name']
            color = COLORS[name]
            tip_data = res['tip_' + comp]
            errors = []

            for step in CHECK_STEPS:
                lf = step / N_STEPS
                ref_interp = ref_full * lf  # linear interpolation (approximate)
                nS = len(tip_data) - 1
                if step <= nS:
                    val_num = tip_data[step]
                else:
                    val_num = 0.0
                if abs(ref_interp) > 1e-10:
                    err = abs(val_num - ref_interp) / abs(ref_interp) * 100.0
                else:
                    err = abs(val_num) * 100.0
                errors.append(min(err, 300.0))

            x_pos = x_base + fi * bar_width
            bars = ax.bar(x_pos, errors, bar_width * 0.9, color=color,
                          alpha=0.85, label=name, edgecolor='white', linewidth=0.3)
            for bar, err in zip(bars, errors):
                if 0.5 < err < 300:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.5,
                            '{:.1f}'.format(err), ha='center', va='bottom',
                            fontsize=6, color=TEXT_COLOR)

        ax.set_xticks(x_base + bar_width * (n_form - 1) / 2)
        ax.set_xticklabels(['LF={:.0f}%'.format(s / N_STEPS * 100)
                            for s in CHECK_STEPS], fontsize=9)
        ax.set_ylabel('{} Error (%)'.format(label))
        ax.set_title('{} Error vs Bathe Ref'.format(label),
                     fontsize=11, color=TEXT_COLOR)
        ax.legend(facecolor='#1a1a24', edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR, fontsize=7, loc='upper left')

    fig.suptitle('Bathe & Bolourchi Curved Cantilever -- Error vs Reference\n'
                 '(LF<100%: linearly interpolated reference, approximate)',
                 fontsize=13, color=TEXT_COLOR, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, 'hex8_curved_error.png')
    fig.savefig(path, dpi=200, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: {}".format(path))


# ================================================================
# Main
# ================================================================

def run_mesh_refinement():
    """Mesh refinement study: TL only, multiple mesh densities."""
    meshes = [
        (12, 1, 1),
        (24, 2, 2),
        (48, 4, 4),
    ]

    print("=" * 70)
    print("Bathe & Bolourchi (1979) -- Hex8 Mesh Refinement Study (TL only)")
    print("  R={}, H={}, W={}, E={:.0e}, nu={}".format(R, H, W, E_VAL, NU_VAL))
    print("  P={} in +Z, {} load steps, UmfPackSOE + RCM".format(P_TOTAL, N_STEPS))
    print("  Reference: ux={}, uy={}, uz={}".format(
        BATHE_REF['ux'], BATHE_REF['uy'], BATHE_REF['uz']))
    print("=" * 70)

    refinement_results = []
    for n_arc, n_rad, n_width in meshes:
        n_elem = n_arc * n_rad * n_width
        mesh_label = "{}x{}x{}".format(n_arc, n_rad, n_width)
        print("\n  Running TL on {} mesh ({} elements)...".format(
            mesh_label, n_elem))

        # Compute DOFs: total nodes * 3 (all DOFs), free DOFs = (total - fixed) * 3
        n_nodes = (n_arc + 1) * (n_rad + 1) * (n_width + 1)
        n_fixed = (n_rad + 1) * (n_width + 1)
        n_dof = n_nodes * 3

        t0 = _time.perf_counter()
        res = run_formulation('TL', TotalLagrangianContinuumKinematics,
                              n_arc=n_arc, n_rad=n_rad, n_width=n_width,
                              system=UmfPackSOE(1), numberer=RCM(1))
        wall = _time.perf_counter() - t0

        ux = res['tip_ux'][-1]
        uy = res['tip_uy'][-1]
        uz = res['tip_uz'][-1]
        status = "CONVERGED" if res['converged'] else "FAILED"
        print("    {} in {:.1f}s  ({} DOFs)".format(status, res['wall_time'], n_dof))
        print("    Tip: ux={:.4f}, uy={:.4f}, uz={:.4f}".format(ux, uy, uz))

        refinement_results.append({
            'mesh': mesh_label,
            'n_arc': n_arc, 'n_rad': n_rad, 'n_width': n_width,
            'n_elem': n_elem,
            'n_dof': n_dof,
            'ux': ux, 'uy': uy, 'uz': uz,
            'converged': res['converged'],
            'wall_time': res['wall_time'],
        })

        # Check if convergence toward reference is clear before running next
        if len(refinement_results) >= 2:
            prev_uz = refinement_results[-2]['uz']
            curr_uz = refinement_results[-1]['uz']
            ref_uz = BATHE_REF['uz']
            # If both are moving toward reference, continue
            if abs(curr_uz - ref_uz) < abs(prev_uz - ref_uz):
                print("    -> Converging toward ref (uz: {:.2f} -> {:.2f}, "
                      "ref={})".format(prev_uz, curr_uz, ref_uz))
            else:
                print("    -> NOT converging toward ref (uz: {:.2f} -> {:.2f}, "
                      "ref={})".format(prev_uz, curr_uz, ref_uz))

        # Skip 48x4x4 if previous run was already slow
        if res['wall_time'] > 300:
            print("    -> Skipping finer meshes (>{:.0f}s)".format(
                res['wall_time']))
            break

    # Summary table
    print("\n" + "=" * 70)
    print("  MESH REFINEMENT SUMMARY (TL, P={})".format(P_TOTAL))
    print("  {:>10s} | {:>8s} | {:>5s} | {:>10s} | {:>10s} | {:>10s} | {:>8s} | {:>6s}".format(
        'Mesh', 'Elements', 'DOFs', 'ux', 'uy', 'uz', 'uz_err%', 'Time'))
    print("  " + "-" * 90)
    for r in refinement_results:
        ref_uz = BATHE_REF['uz']
        uz_err = abs(r['uz'] - ref_uz) / abs(ref_uz) * 100.0
        print("  {:>10s} | {:>8d} | {:>5d} | {:>10.4f} | {:>10.4f} | {:>10.4f} | {:>7.1f}% | {:>5.1f}s".format(
            r['mesh'], r['n_elem'], r['n_dof'], r['ux'], r['uy'], r['uz'],
            uz_err, r['wall_time']))
    print("  {:>10s} | {:>8s} | {:>5s} | {:>10.4f} | {:>10.4f} | {:>10.4f} | {:>7.1f}% |".format(
        'Bathe ref', '—', '—',
        BATHE_REF['ux'], BATHE_REF['uy'], BATHE_REF['uz'], 0.0))
    print("=" * 70)

    # Plot: uz vs number of elements (log-log)
    if len(refinement_results) >= 2:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        style_dark(fig, [ax])

        n_elems = [r['n_elem'] for r in refinement_results]
        uz_vals = [r['uz'] for r in refinement_results]

        ax.plot(n_elems, uz_vals, 'o-', color=COLORS['TL'], linewidth=2,
                markersize=8, label='TL Hex8', zorder=5)
        ax.axhline(y=BATHE_REF['uz'], color=ANALYTICAL_COLOR, linestyle='--',
                   linewidth=1.5, label='Bathe ref = {}'.format(BATHE_REF['uz']),
                   zorder=3)

        for r in refinement_results:
            ax.annotate(r['mesh'], (r['n_elem'], r['uz']),
                        textcoords="offset points", xytext=(8, -12),
                        fontsize=9, color=TEXT_COLOR)

        ax.set_xscale('log')
        ax.set_xlabel('Number of Elements', fontsize=11)
        ax.set_ylabel(r'Tip $u_z$', fontsize=11)
        ax.set_title('Hex8 Mesh Convergence -- Bathe & Bolourchi Curved Cantilever',
                     fontsize=12, color=TEXT_COLOR, fontweight='bold')
        ax.legend(facecolor='#1a1a24', edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR, fontsize=10)

        fig.tight_layout()
        path = os.path.join(SAVE_DIR, 'hex8_curved_mesh_convergence.png')
        fig.savefig(path, dpi=200, facecolor=BG_COLOR, bbox_inches='tight')
        plt.close(fig)
        print("\n  Saved: {}".format(path))

    return refinement_results


def run_comparison_study():
    """Original 4-formulation comparison on 12x1x1 mesh."""
    print("=" * 70)
    print("Bathe & Bolourchi (1979) Curved Cantilever -- Hex8 Comparison Study")
    print("  R={}, H={}, W={}, E={:.0e}, nu={}".format(R, H, W, E_VAL, NU_VAL))
    print("  P={} in +Z (out of arch plane), {} load steps".format(P_TOTAL, N_STEPS))
    print("  Mesh: {}x1x1 Hex8 ({} elements)".format(N_ARC, N_ARC))
    print("  Reference: ux={}, uy={}, uz={}".format(
        BATHE_REF['ux'], BATHE_REF['uy'], BATHE_REF['uz']))
    print("=" * 70)

    # Verify mesh geometry
    node_coords, elem_conn, free_nids, fixed_nids = build_curved_mesh()
    print("\n  Mesh geometry check:")
    for nid in fixed_nids:
        print("    Fixed node {}: ({:.2f}, {:.2f}, {:.2f})".format(nid, *node_coords[nid]))
    for nid in free_nids:
        print("    Free  node {}: ({:.2f}, {:.2f}, {:.2f})".format(nid, *node_coords[nid]))

    all_results = []
    for name, kin_class in FORMULATIONS:
        print("\n  Running {}...".format(name))
        res = run_formulation(name, kin_class)
        all_results.append(res)
        status = "CONVERGED" if res['converged'] else "FAILED"
        ux = res['tip_ux'][-1]
        uy = res['tip_uy'][-1]
        uz = res['tip_uz'][-1]
        print("    {} -- {} in {:.2f}s".format(name, status, res['wall_time']))
        print("    Tip: ux={:.4f}, uy={:.4f}, uz={:.4f}".format(ux, uy, uz))

    # Error table vs Bathe at full load
    print("\n" + "-" * 70)
    print("  Error vs Bathe reference at full load (P={})".format(P_TOTAL))
    print("  {:>10s}  {:>12s}  {:>12s}  {:>12s}".format(
        'Form.', 'ux err%', 'uy err%', 'uz err%'))
    print("  " + "-" * 50)
    for res in all_results:
        errs = []
        for comp in ['ux', 'uy', 'uz']:
            ref = BATHE_REF[comp]
            val = res['tip_' + comp][-1]
            if abs(ref) > 1e-10:
                errs.append(abs(val - ref) / abs(ref) * 100.0)
            else:
                errs.append(abs(val) * 100.0)
        print("  {:>10s}  {:>11.2f}%  {:>11.2f}%  {:>11.2f}%".format(
            res['name'], *errs))

    # TL vs UL agreement
    print("\n  TL vs UL agreement:")
    tl_res = next(r for r in all_results if r['name'] == 'TL')
    ul_res = next(r for r in all_results if r['name'] == 'UL')
    for step in CHECK_STEPS:
        n_tl = len(tl_res['tip_ux']) - 1
        n_ul = len(ul_res['tip_ux']) - 1
        if step <= min(n_tl, n_ul):
            max_diff = max(
                abs(tl_res['tip_ux'][step] - ul_res['tip_ux'][step]),
                abs(tl_res['tip_uy'][step] - ul_res['tip_uy'][step]),
                abs(tl_res['tip_uz'][step] - ul_res['tip_uz'][step]),
            )
            status = "PASS" if max_diff < 1e-6 else "FAIL"
            print("    Step {:2d}: max|TL-UL| = {:.2e}  {}".format(
                step, max_diff, status))

    # Wall times
    print("\n  Wall times:")
    for res in all_results:
        print("    {:>8s}: {:.2f}s".format(res['name'], res['wall_time']))

    # Generate figures
    print("\n  Generating figures...")
    plot_deformed(all_results)
    plot_load_disp(all_results)
    plot_convergence(all_results)
    plot_error(all_results)

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Bathe & Bolourchi Curved Cantilever -- Hex8')
    parser.add_argument('--refine', action='store_true',
                        help='Run mesh refinement study (TL only)')
    args = parser.parse_args()

    if args.refine:
        run_mesh_refinement()
    else:
        run_comparison_study()
