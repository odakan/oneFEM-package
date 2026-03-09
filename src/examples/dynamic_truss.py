# Dynamic benchmark: SDOF-equivalent truss under step load
# Verifies Newmark average acceleration (gamma=0.5, beta=0.25)
# against the closed-form solution for an undamped SDOF system.
#
# Analytical solution for step load F0 on undamped SDOF:
#   u(t) = (F0/k) * [1 - cos(omega*t)]
#   v(t) = (F0/k) * omega * sin(omega*t)
#   a(t) = (F0/k) * omega^2 * cos(omega*t)
#   omega = sqrt(k/m)
#
# Model: single truss element, one node fixed, one free in x-direction
#
#   Fixed o==========o Free  -->  F0 (step load in x)
#   nd1 [0,0,0]      nd2 [L,0,0]
#

import numpy as np

# import domain
from oneFEM.model import Domain
# modeling tools
from oneFEM.model.element.truss import Truss
from oneFEM.model.node import Node36
from oneFEM.model.element.section import Rectangular
from oneFEM.model.material.uniaxial import Elastic
from oneFEM.model.tseries import Constant
from oneFEM.model.pattern import Plain as PlainPattern
# result tools
from oneFEM.output.recorder import NodeRecorder
# import analysis
from oneFEM.analysis import Analysis
# analysis tools
from oneFEM.analysis.algorithm import Linear, Newton
from oneFEM.analysis.constraints import Plain as PlainConstraints
from oneFEM.analysis.numberer import Plain as PlainNumberer
from oneFEM.analysis.system import FullGeneral, UmfPackSOE
from oneFEM.analysis.integrator import Newmark
from oneFEM.analysis.test import NormUnbalance
# import simulation manager
from oneFEM import SimulationManager

# ============================================================
# Model parameters
# ============================================================
E = 2.0e9       # Pa
A = 0.02        # m^2 (h=0.2, w=0.1)
L = 2.0         # m
rho_bar = 100.0 # mass per unit length (kg/m)
F0 = 1000.0     # Step load (N)

# Derived quantities
EA = E * A
k = EA / L                   # Spring stiffness
m_eff = rho_bar * L / 2.0    # Lumped mass at free node
omega = np.sqrt(k / m_eff)
T = 2.0 * np.pi / omega

# Use 500 steps/period for 1 period to keep Newmark period elongation < 1e-4
nSteps = 500
dt = T / nSteps

print(f"k = {k:.2e} N/m")
print(f"m_eff = {m_eff:.2e} kg")
print(f"omega = {omega:.4f} rad/s")
print(f"T = {T:.6f} s")
print(f"dt = {dt:.6e} s")
print(f"nSteps = {nSteps}")
print()


def run_dynamic_analysis(alg_name, sys_name):
    """Run dynamic analysis with specified algorithm and system solver."""
    # Initialize domain
    model = Domain(nD=3)

    # Nodes: nd1 fully fixed, nd2 free in x only
    nd1 = Node36(1, coord=[0.0, 0.0, 0.0], fix=[1, 1, 1, 1, 1, 1])
    nd2 = Node36(2, coord=[L, 0.0, 0.0],   fix=[0, 1, 1, 1, 1, 1])
    model.add(nd1, nd2)

    # Material and section
    steel = Elastic(1, E=E)
    section = Rectangular(1, mat=steel, h=0.2, w=0.1)

    # Truss with mass (lumped)
    tr1 = Truss(1, nodes=[nd1, nd2], section=section, rho=rho_bar, cMass=False)
    model.add(tr1)

    # Constant time series (step load: constant from t=0)
    ts = Constant(1, factor=1.0)

    # Load pattern: F0 in x on node 2
    load_data = [[2, F0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    pattern = PlainPattern(1, tseries=ts, load=load_data)
    model.add(pattern)

    # Recorder for node 2 displacement, velocity, acceleration
    rec_disp = NodeRecorder(1, nd2, dofs=[1], results=['displacement'])
    rec_vel = NodeRecorder(2, nd2, dofs=[1], results=['velocity'])
    rec_accel = NodeRecorder(3, nd2, dofs=[1], results=['acceleration'])
    model.add(rec_disp, rec_vel, rec_accel)

    # Algorithm
    if alg_name == 'Linear':
        alg = Linear(1)
    else:
        alg = Newton(1)

    # System
    if sys_name == 'FullGeneral':
        syst = FullGeneral(1)
    else:
        syst = UmfPackSOE(1)

    # Other analysis components
    const = PlainConstraints(1)
    numb = PlainNumberer(1)
    integ = Newmark(1, gamma=0.5, beta=0.25)
    ctest = NormUnbalance(1, tol=1e-8, maxIter=10)

    analysis = Analysis(1, algorithm=alg, constraints=const,
                        integrator=integ, system=syst, test=ctest,
                        numberer=numb)

    sim = SimulationManager(1, model, analysis, dt=dt)
    sim.analyze(nSteps, dt=dt)

    # Extract recorded data
    disp_history = [v[0] if hasattr(v, '__getitem__') else v
                    for v in rec_disp.data['displacement']]
    vel_history = [v[0] if hasattr(v, '__getitem__') else v
                   for v in rec_vel.data['velocity']]
    accel_history = [v[0] if hasattr(v, '__getitem__') else v
                     for v in rec_accel.data['acceleration']]

    return disp_history, vel_history, accel_history


# ============================================================
# Run with different algorithm/system combinations
# ============================================================
configs = [
    ('Linear', 'FullGeneral'),
    ('Newton', 'FullGeneral'),
    ('Linear', 'UmfPackSOE'),
]

results = {}
for alg_name, sys_name in configs:
    print(f"--- Running: {alg_name} + {sys_name} ---")
    d, v, a = run_dynamic_analysis(alg_name, sys_name)
    results[(alg_name, sys_name)] = (d, v, a)
    print()


# ============================================================
# Analytical solution
# ============================================================
u_static = F0 / k

time_arr = np.arange(1, nSteps + 1) * dt
u_exact = u_static * (1.0 - np.cos(omega * time_arr))
v_exact = u_static * omega * np.sin(omega * time_arr)
a_exact = u_static * omega**2 * np.cos(omega * time_arr)

# ============================================================
# Verification
# ============================================================
print("\n========== DYNAMIC TRUSS BENCHMARK RESULTS ==========")
print(f"  E = {E:.2e} Pa,  A = {A} m^2,  L = {L} m")
print(f"  rho = {rho_bar} kg/m,  m_eff = {m_eff} kg")
print(f"  k = {k:.2e} N/m,  omega = {omega:.4f} rad/s,  T = {T:.6f} s")
print(f"  F0 = {F0} N,  u_static = {u_static:.6e} m")
print(f"  dt = {dt:.6e} s,  nSteps = {nSteps}")
print()

tol = 1e-4  # relative error tolerance
all_pass = True

for (alg_name, sys_name), (d_hist, v_hist, a_hist) in results.items():
    print(f"  Config: {alg_name} + {sys_name}")

    d_arr = np.array(d_hist)
    v_arr = np.array(v_hist)
    a_arr = np.array(a_hist)

    # Relative error for displacement
    max_u = np.max(np.abs(u_exact))
    err_u = np.max(np.abs(d_arr - u_exact)) / max_u
    pass_u = err_u < tol

    # Relative error for velocity
    max_v = np.max(np.abs(v_exact))
    err_v = np.max(np.abs(v_arr - v_exact)) / max_v
    pass_v = err_v < tol

    # Relative error for acceleration
    max_a = np.max(np.abs(a_exact))
    err_a = np.max(np.abs(a_arr - a_exact)) / max_a
    pass_a = err_a < tol

    print(f"    Displacement: max rel error = {err_u:.6e}  {'PASS' if pass_u else 'FAIL'}")
    print(f"    Velocity:     max rel error = {err_v:.6e}  {'PASS' if pass_v else 'FAIL'}")
    print(f"    Acceleration: max rel error = {err_a:.6e}  {'PASS' if pass_a else 'FAIL'}")

    if not (pass_u and pass_v and pass_a):
        all_pass = False
    print()

# Cross-check: Linear and Newton should give identical results
d_lin, v_lin, a_lin = results[('Linear', 'FullGeneral')]
d_nwt, v_nwt, a_nwt = results[('Newton', 'FullGeneral')]
cross_err = np.max(np.abs(np.array(d_lin) - np.array(d_nwt)))
pass_cross = cross_err < 1e-12
print(f"  Cross-check (Linear vs Newton): max diff = {cross_err:.6e}  {'PASS' if pass_cross else 'FAIL'}")
if not pass_cross:
    all_pass = False

# Cross-check: FullGeneral and UmfPackSOE should give identical results
d_fg, v_fg, a_fg = results[('Linear', 'FullGeneral')]
d_um, v_um, a_um = results[('Linear', 'UmfPackSOE')]
cross_err2 = np.max(np.abs(np.array(d_fg) - np.array(d_um)))
pass_cross2 = cross_err2 < 1e-12
print(f"  Cross-check (FullGeneral vs UmfPackSOE): max diff = {cross_err2:.6e}  {'PASS' if pass_cross2 else 'FAIL'}")
if not pass_cross2:
    all_pass = False

print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
print("=====================================================\n")

# ============================================================
# Plots
# ============================================================
import matplotlib.pyplot as plt

# Use the Linear + FullGeneral result for plotting
d_num, v_num, a_num = results[('Linear', 'FullGeneral')]
d_num = np.array(d_num)
v_num = np.array(v_num)
a_num = np.array(a_num)

# Time in milliseconds for readability
time_ms = time_arr * 1e3
T_ms = T * 1e3

# ---------------------------------------------------------------
#  Figure 1 — Full response: numerical (dots) vs analytical (line)
# ---------------------------------------------------------------
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
fig1.suptitle(
    'SDOF Truss — Newmark vs Closed-Form Solution\n'
    f'Step load F = {F0:.0f} N,  k = {k:.0e} N/m,  m = {m_eff:.0f} kg,  '
    f'T = {T_ms:.3f} ms',
    fontsize=13, fontweight='bold',
)

# Displacement
ax1.plot(time_ms, u_exact * 1e6, '-', color='black', lw=1.5,
         label=r'Exact: $u = (F/k)[1 - \cos(\omega t)]$')
ax1.plot(time_ms, d_num * 1e6, 'o', color='#e74c3c', ms=2.0, markevery=10,
         label='oneFEM (Newmark, 500 steps)')
ax1.set_ylabel(r'Displacement [$\mu$m]', fontsize=11)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.25)

# Velocity
ax2.plot(time_ms, v_exact * 1e3, '-', color='black', lw=1.5,
         label=r'Exact: $\dot{u} = (F/k)\,\omega\,\sin(\omega t)$')
ax2.plot(time_ms, v_num * 1e3, 'o', color='#e74c3c', ms=2.0, markevery=10,
         label='oneFEM (Newmark, 500 steps)')
ax2.set_ylabel('Velocity [mm/s]', fontsize=11)
ax2.legend(fontsize=10, loc='upper left')
ax2.grid(True, alpha=0.25)

# Acceleration
ax3.plot(time_ms, a_exact, '-', color='black', lw=1.5,
         label=r'Exact: $\ddot{u} = (F/k)\,\omega^2\cos(\omega t)$')
ax3.plot(time_ms, a_num, 'o', color='#e74c3c', ms=2.0, markevery=10,
         label='oneFEM (Newmark, 500 steps)')
ax3.set_ylabel('Acceleration [m/s$^2$]', fontsize=11)
ax3.set_xlabel('Time [ms]', fontsize=11)
ax3.legend(fontsize=10, loc='upper left')
ax3.grid(True, alpha=0.25)

fig1.tight_layout(rect=[0, 0, 1, 0.93])
fig1.savefig('dynamic_truss_response.png', dpi=180, bbox_inches='tight')
print("Figure 1 saved to: dynamic_truss_response.png")

# ---------------------------------------------------------------
#  Figure 2 — Zoomed view near peak to show the tiny difference
# ---------------------------------------------------------------
# Zoom into the region around the displacement peak (t/T ~ 0.5)
i_peak = np.argmax(u_exact)
# Take a window of ±30 steps around the peak
win = 30
i0 = max(0, i_peak - win)
i1 = min(len(time_ms), i_peak + win)
t_zoom = time_ms[i0:i1]

fig2, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(14, 4.5))
fig2.suptitle(
    'Zoomed Near Peak — Can You See the Difference?',
    fontsize=13, fontweight='bold',
)

# Zoomed displacement
ax4.plot(t_zoom, u_exact[i0:i1] * 1e6, '-', color='black', lw=2, label='Exact')
ax4.plot(t_zoom, d_num[i0:i1] * 1e6, 'o-', color='#e74c3c', lw=1, ms=3, label='Newmark')
ax4.set_xlabel('Time [ms]', fontsize=10)
ax4.set_ylabel(r'Displacement [$\mu$m]', fontsize=10)
ax4.set_title('Displacement near peak', fontsize=11)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.25)

# Zoomed velocity
ax5.plot(t_zoom, v_exact[i0:i1] * 1e3, '-', color='black', lw=2, label='Exact')
ax5.plot(t_zoom, v_num[i0:i1] * 1e3, 'o-', color='#e74c3c', lw=1, ms=3, label='Newmark')
ax5.set_xlabel('Time [ms]', fontsize=10)
ax5.set_ylabel('Velocity [mm/s]', fontsize=10)
ax5.set_title('Velocity near peak', fontsize=11)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.25)

# Zoomed acceleration
ax6.plot(t_zoom, a_exact[i0:i1], '-', color='black', lw=2, label='Exact')
ax6.plot(t_zoom, a_num[i0:i1], 'o-', color='#e74c3c', lw=1, ms=3, label='Newmark')
ax6.set_xlabel('Time [ms]', fontsize=10)
ax6.set_ylabel('Acceleration [m/s$^2$]', fontsize=10)
ax6.set_title('Acceleration near peak', fontsize=11)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.25)

fig2.tight_layout(rect=[0, 0, 1, 0.91])
fig2.savefig('dynamic_truss_zoomed.png', dpi=180, bbox_inches='tight')
print("Figure 2 saved to: dynamic_truss_zoomed.png")

# ---------------------------------------------------------------
#  Figure 3 — Difference (numerical minus analytical) over time
# ---------------------------------------------------------------
fig3, (ax7, ax8, ax9) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig3.suptitle(
    'Error = Newmark $-$ Exact\n'
    'How much the numerical solution deviates from the analytical one',
    fontsize=13, fontweight='bold',
)

ax7.plot(time_ms, (d_num - u_exact) * 1e9, color='#3498db', lw=1.0)
ax7.set_ylabel(r'$\Delta u$ [nm]', fontsize=11)
ax7.set_title(
    f'Displacement error  (max = {np.max(np.abs(d_num - u_exact))*1e9:.2f} nm '
    + r'out of ' + f'{np.max(np.abs(u_exact))*1e6:.1f}' + r' $\mu$m)',
    fontsize=10)
ax7.axhline(0, color='grey', lw=0.5)
ax7.grid(True, alpha=0.25)

ax8.plot(time_ms, (v_num - v_exact) * 1e6, color='#3498db', lw=1.0)
ax8.set_ylabel(r'$\Delta \dot{u}$ [$\mu$m/s]', fontsize=11)
ax8.set_title(
    f'Velocity error  (max = {np.max(np.abs(v_num - v_exact))*1e6:.2f}' + r' $\mu$m/s '
    + f'out of {np.max(np.abs(v_exact))*1e3:.2f} mm/s)',
    fontsize=10)
ax8.axhline(0, color='grey', lw=0.5)
ax8.grid(True, alpha=0.25)

ax9.plot(time_ms, (a_num - a_exact) * 1e3, color='#3498db', lw=1.0)
ax9.set_ylabel(r'$\Delta \ddot{u}$ [mm/s$^2$]', fontsize=11)
ax9.set_xlabel('Time [ms]', fontsize=11)
ax9.set_title(
    f'Acceleration error  (max = {np.max(np.abs(a_num - a_exact))*1e3:.3f} mm/s$^2$ '
    f'out of {np.max(np.abs(a_exact)):.2f} m/s$^2$)',
    fontsize=10)
ax9.axhline(0, color='grey', lw=0.5)
ax9.grid(True, alpha=0.25)

fig3.tight_layout(rect=[0, 0, 1, 0.92])
fig3.savefig('dynamic_truss_error.png', dpi=180, bbox_inches='tight')
print("Figure 3 saved to: dynamic_truss_error.png")

plt.show()
