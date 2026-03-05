# Dynamic truss under earthquake ground motion (uniform excitation)
#
# OpenSees-style UniformExcitation applies ground acceleration directly:
#   M * a_rel + C * v_rel + K * u_rel = -M * r * a_g(t)
#
# Record: rec_000_H1_ACC.txt, dt = 0.01 s, direction = X (DOF 1)
#
# Model: single truss element, one fixed node, one free in x
#
#   Fixed o==========o Free
#   nd1 [0,0,0]      nd2 [L,0,0]
#

import numpy as np

# oneFEM imports
from oneFEM.model import Domain
from oneFEM.model.element.truss import Truss
from oneFEM.model.node import Node36
from oneFEM.model.element.section import Rectangular
from oneFEM.model.material.uniaxial import Elastic
from oneFEM.model.tseries import Path
from oneFEM.model.pattern import UniformExcitation
from oneFEM.output.recorder import NodeRecorder
from oneFEM.analysis import Analysis
from oneFEM.analysis.algorithm import Linear
from oneFEM.analysis.constraints import Plain as PlainConstraints
from oneFEM.analysis.numberer import Plain as PlainNumberer
from oneFEM.analysis.system import FullGeneral
from oneFEM.analysis.integrator import Newmark
from oneFEM.analysis.test import NormUnbalance
from oneFEM import SimulationManager

# ============================================================
# Load ground motion record
# ============================================================
acc_data = np.loadtxt('rec_000_H1_ACC.txt')
dt_record = 0.01  # s
nSteps = len(acc_data)
t_record = np.arange(nSteps) * dt_record

print(f"Ground motion: {nSteps} points, dt = {dt_record} s, "
      f"duration = {(nSteps-1)*dt_record:.2f} s")
print(f"PGA = {np.max(np.abs(acc_data)):.4f} m/s^2")
print()

# ============================================================
# Model parameters
# ============================================================
E = 2.0e9        # Pa
A = 0.02         # m^2
L = 2.0          # m
rho_bar = 100.0  # kg/m

k = E * A / L
m_eff = rho_bar * L / 2.0
omega = np.sqrt(k / m_eff)
T_n = 2.0 * np.pi / omega
f_hz = 1.0 / T_n

print(f"k = {k:.2e} N/m,  m = {m_eff:.0f} kg")
print(f"T = {T_n*1e3:.3f} ms  ({f_hz:.0f} Hz),  omega = {omega:.1f} rad/s")
print(f"dt/T = {dt_record/T_n:.1f}")
print()

# ============================================================
# Build oneFEM model
# ============================================================
model = Domain(nD=3)

nd1 = Node36(1, coord=[0.0, 0.0, 0.0], fix=[1, 1, 1, 1, 1, 1])
nd2 = Node36(2, coord=[L, 0.0, 0.0],   fix=[0, 1, 1, 1, 1, 1])
model.add(nd1, nd2)

steel = Elastic(1, E=E)
section = Rectangular(1, mat=steel, h=0.2, w=0.1)
tr1 = Truss(1, nodes=[nd1, nd2], section=section, rho=rho_bar, cMass=False)
model.add(tr1)

# Ground acceleration time series
ts_gm = Path(1, dt=dt_record, data_values=acc_data.tolist(), factor=1.0)

# UniformExcitation in X direction (DOF 1)
eq_pattern = UniformExcitation(1, direction=1, tseries=ts_gm)
model.add(eq_pattern)

# Recorders
rec_disp = NodeRecorder(1, nd2, dofs=[1], results=['displacement'])
rec_vel = NodeRecorder(2, nd2, dofs=[1], results=['velocity'])
rec_accel = NodeRecorder(3, nd2, dofs=[1], results=['acceleration'])
model.add(rec_disp, rec_vel, rec_accel)

# Analysis
alg = Linear(1)
const = PlainConstraints(1)
numb = PlainNumberer(1)
syst = FullGeneral(1)
integ = Newmark(1, gamma=0.5, beta=0.25)
ctest = NormUnbalance(1, tol=1e-8, maxIter=10)

analysis = Analysis(1, algorithm=alg, constraints=const,
                    integrator=integ, system=syst, test=ctest,
                    numberer=numb)

sim = SimulationManager(1, model, analysis, dt=dt_record)

print(f"Running {nSteps} steps ...")
sim.analyze(nSteps, dt=dt_record)
print()

# ============================================================
# Extract results
# ============================================================
u_hist = np.array([v[0] if hasattr(v, '__getitem__') else v
                   for v in rec_disp.data['displacement']])
v_hist = np.array([v[0] if hasattr(v, '__getitem__') else v
                   for v in rec_vel.data['velocity']])
a_hist = np.array([v[0] if hasattr(v, '__getitem__') else v
                   for v in rec_accel.data['acceleration']])

time_arr = np.arange(1, nSteps + 1) * dt_record

# ============================================================
# Reference: direct SDOF Newmark integration
# m*a + k*u = -m*a_g(t)
# ============================================================
gamma = 0.5
beta = 0.25
dt = dt_record

u_ref = np.zeros(nSteps + 1)
v_ref = np.zeros(nSteps + 1)
a_ref = np.zeros(nSteps + 1)

# Initial acceleration: a_0 = -a_g(0)  (since u=v=0)
a_ref[0] = -acc_data[0]

c3 = 1.0 / (beta * dt * dt)
c2 = gamma / (beta * dt)

for i in range(nSteps):
    # Force at step i+1 (Newmark uses F at t+dt)
    # Use i+1 if available, else hold last value
    idx = min(i + 1, nSteps - 1)
    F_ext = -m_eff * acc_data[idx]

    k_eff = k + c3 * m_eff
    rhs = F_ext + m_eff * (c3 * u_ref[i]
                           + (1.0 / (beta * dt)) * v_ref[i]
                           + (1.0 / (2.0 * beta) - 1.0) * a_ref[i])

    u_ref[i + 1] = rhs / k_eff
    a_ref[i + 1] = (c3 * (u_ref[i + 1] - u_ref[i])
                     - (1.0 / (beta * dt)) * v_ref[i]
                     - (1.0 / (2.0 * beta) - 1.0) * a_ref[i])
    v_ref[i + 1] = (v_ref[i]
                     + dt * ((1.0 - gamma) * a_ref[i] + gamma * a_ref[i + 1]))

# Skip initial condition to align with recorder output
u_ref = u_ref[1:]
v_ref = v_ref[1:]
a_ref = a_ref[1:]

# ============================================================
# Verification
# ============================================================
max_u_ref = np.max(np.abs(u_ref))
max_v_ref = np.max(np.abs(v_ref))
max_a_ref = np.max(np.abs(a_ref))

err_u = np.max(np.abs(u_hist - u_ref)) / max_u_ref if max_u_ref > 0 else 0
err_v = np.max(np.abs(v_hist - v_ref)) / max_v_ref if max_v_ref > 0 else 0
err_a = np.max(np.abs(a_hist - a_ref)) / max_a_ref if max_a_ref > 0 else 0

print("========== EARTHQUAKE BENCHMARK ==========")
print(f"  Max rel. displacement: {np.max(np.abs(u_hist))*1e6:.4f} um")
print(f"  Max rel. velocity:     {np.max(np.abs(v_hist))*1e3:.4f} mm/s")
print(f"  Max rel. acceleration: {np.max(np.abs(a_hist)):.4f} m/s^2")
print()
print(f"  vs SDOF Newmark reference:")
print(f"    Disp  rel error: {err_u:.2e}  {'PASS' if err_u < 1e-6 else 'CHECK'}")
print(f"    Vel   rel error: {err_v:.2e}  {'PASS' if err_v < 1e-6 else 'CHECK'}")
print(f"    Accel rel error: {err_a:.2e}  {'PASS' if err_a < 1e-6 else 'CHECK'}")
print("==========================================\n")

# ============================================================
# Plots
# ============================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
fig.suptitle(
    'Earthquake Response — SDOF Truss under Uniform Ground Acceleration\n'
    r'Record: rec_000_H1_ACC.txt (X-dir),  $\Delta t$ = '
    f'{dt_record} s,  T = {T_n*1e3:.3f} ms ({f_hz:.0f} Hz)',
    fontsize=13, fontweight='bold',
)

# --- Ground motion input ---
ax = axes[0]
ax.plot(t_record, acc_data, color='#7f8c8d', lw=0.5)
ax.set_ylabel(r'$a_g$ [m/s$^2$]', fontsize=11)
ax.set_title('Input Ground Acceleration', fontsize=11)
ax.grid(True, alpha=0.25)
ax.annotate(f'PGA = {np.max(np.abs(acc_data)):.3f} m/s$^2$',
            xy=(0.98, 0.92), xycoords='axes fraction', ha='right', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

# --- Displacement ---
ax = axes[1]
ax.plot(time_arr, u_ref * 1e6, '-', color='black', lw=1.2,
        label='SDOF Newmark (reference)')
ax.plot(time_arr, u_hist * 1e6, '--', color='#e74c3c', lw=0.9,
        label='oneFEM', dashes=(6, 3))
ax.set_ylabel(r'$u_{rel}$ [$\mu$m]', fontsize=11)
ax.set_title('Relative Displacement', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.25)

# --- Velocity ---
ax = axes[2]
ax.plot(time_arr, v_ref * 1e3, '-', color='black', lw=1.2,
        label='SDOF Newmark (reference)')
ax.plot(time_arr, v_hist * 1e3, '--', color='#e74c3c', lw=0.9,
        label='oneFEM', dashes=(6, 3))
ax.set_ylabel(r'$\dot{u}_{rel}$ [mm/s]', fontsize=11)
ax.set_title('Relative Velocity', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.25)

# --- Acceleration ---
ax = axes[3]
ax.plot(time_arr, a_ref, '-', color='black', lw=1.2,
        label='SDOF Newmark (reference)')
ax.plot(time_arr, a_hist, '--', color='#e74c3c', lw=0.9,
        label='oneFEM', dashes=(6, 3))
ax.set_ylabel(r'$\ddot{u}_{rel}$ [m/s$^2$]', fontsize=11)
ax.set_xlabel('Time [s]', fontsize=11)
ax.set_title('Relative Acceleration', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.25)

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('dynamic_truss_earthquake.png', dpi=180, bbox_inches='tight')
print("Figure saved to: dynamic_truss_earthquake.png")
plt.show()
