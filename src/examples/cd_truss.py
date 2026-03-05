# Explicit dynamic benchmark: CentralDifference + Linear
#
# SDOF elastic truss under step load.
# Verifies CentralDifference against closed-form solution.
#
# Analytical solution for step load F0 on undamped SDOF:
#   u(t) = (F0/k) * [1 - cos(omega*t)]
#
# Central Difference requires dt < T/pi for stability.
# We use dt = T/100 (well within stability limit).
#
# Model: single truss, one fixed node, one free in x
#   Fixed o==========o Free  -->  F0 (step load)
#   nd1 [0,0,0]      nd2 [L,0,0]
#

import numpy as np

# oneFEM imports
from oneFEM.model import Domain
from oneFEM.model.element.truss import Truss
from oneFEM.model.node import Node36
from oneFEM.model.element.section import Rectangular
from oneFEM.model.material.uniaxial import Elastic
from oneFEM.model.tseries import Constant
from oneFEM.model.pattern import Plain as PlainPattern
from oneFEM.output.recorder import NodeRecorder
from oneFEM.analysis import Analysis
from oneFEM.analysis.algorithm import Linear
from oneFEM.analysis.constraints import Plain as PlainConstraints
from oneFEM.analysis.numberer import Plain as PlainNumberer
from oneFEM.analysis.system import FullGeneral
from oneFEM.analysis.integrator import CDiff
from oneFEM import SimulationManager

# ============================================================
# Model parameters
# ============================================================
E = 2.0e9       # Pa
A = 0.02        # m^2 (h=0.2, w=0.1)
L = 2.0         # m
rho_bar = 100.0 # mass per unit length (kg/m)
F0 = 1000.0     # Step load (N)

# Derived
EA = E * A
k = EA / L
m_eff = rho_bar * L / 2.0
omega = np.sqrt(k / m_eff)
T = 2.0 * np.pi / omega

# CD stability: dt_crit = T / pi
dt_crit = T / np.pi
# Use dt = T/100 (well within stability)
nSteps_per_period = 100
dt = T / nSteps_per_period
nPeriods = 2
nSteps = nSteps_per_period * nPeriods

print(f"k = {k:.2e} N/m")
print(f"m_eff = {m_eff:.2e} kg")
print(f"omega = {omega:.4f} rad/s")
print(f"T = {T:.6f} s")
print(f"dt = {dt:.6e} s  (dt_crit = {dt_crit:.6e} s, ratio = {dt/dt_crit:.3f})")
print(f"nSteps = {nSteps} ({nPeriods} periods)")
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

ts = Constant(1, factor=1.0)
load_data = [[2, F0, 0.0, 0.0, 0.0, 0.0, 0.0]]
pattern = PlainPattern(1, tseries=ts, load=load_data)
model.add(pattern)

rec_disp = NodeRecorder(1, nd2, dofs=[1], results=['displacement'])
rec_vel = NodeRecorder(2, nd2, dofs=[1], results=['velocity'])
rec_accel = NodeRecorder(3, nd2, dofs=[1], results=['acceleration'])
model.add(rec_disp, rec_vel, rec_accel)

# CentralDifference + Linear
alg = Linear(1)
const = PlainConstraints(1)
numb = PlainNumberer(1)
syst = FullGeneral(1)
integ = CDiff(1)

analysis = Analysis(1, algorithm=alg, constraints=const,
                    integrator=integ, system=syst,
                    numberer=numb)

sim = SimulationManager(1, model, analysis, dt=dt)

print(f"Running {nSteps} steps (CentralDifference + Linear) ...")
sim.analyze(nSteps, dt=dt)
print()

# ============================================================
# Extract results
# ============================================================
d_hist = np.array([v[0] if hasattr(v, '__getitem__') else v
                   for v in rec_disp.data['displacement']])
v_hist = np.array([v[0] if hasattr(v, '__getitem__') else v
                   for v in rec_vel.data['velocity']])
a_hist = np.array([v[0] if hasattr(v, '__getitem__') else v
                   for v in rec_accel.data['acceleration']])

time_arr = np.arange(1, nSteps + 1) * dt

# ============================================================
# Analytical solution
# ============================================================
u_static = F0 / k
u_exact = u_static * (1.0 - np.cos(omega * time_arr))
v_exact = u_static * omega * np.sin(omega * time_arr)
a_exact = u_static * omega**2 * np.cos(omega * time_arr)

# ============================================================
# Reference: direct SDOF central difference
# ============================================================
u_cd = np.zeros(nSteps + 2)  # u[-1], u[0], u[1], ...
v_cd = np.zeros(nSteps + 1)
a_cd = np.zeros(nSteps + 1)

# Initial acceleration: a_0 = (F0 - k*u_0) / m
a_cd[0] = F0 / m_eff
# u_{-1} = u_0 - dt*v_0 + 0.5*dt^2*a_0
u_cd[0] = 0.0 - dt * 0.0 + 0.5 * dt * dt * a_cd[0]  # u_{-1}
u_cd[1] = 0.0  # u_0

m_hat = m_eff / (dt * dt)
for i in range(nSteps):
    # u_{n+1} = (F - k*u_n + m_hat*(2*u_n - u_{n-1})) / m_hat
    rhs = F0 - k * u_cd[i + 1] + m_hat * (2.0 * u_cd[i + 1] - u_cd[i])
    u_cd[i + 2] = rhs / m_hat

    # v = (u_{n+1} - u_{n-1}) / (2*dt)
    v_cd[i + 1] = (u_cd[i + 2] - u_cd[i]) / (2.0 * dt) if i < nSteps else 0.0
    # a = (u_{n+1} - 2*u_n + u_{n-1}) / dt^2
    a_cd[i + 1] = (u_cd[i + 2] - 2.0 * u_cd[i + 1] + u_cd[i]) / (dt * dt) if i < nSteps else 0.0

u_cd_out = u_cd[2:]  # u_1, u_2, ...
v_cd_out = v_cd[1:]
a_cd_out = a_cd[1:]

# ============================================================
# Verification
# ============================================================
max_u_exact = np.max(np.abs(u_exact))
max_v_exact = np.max(np.abs(v_exact))
max_a_exact = np.max(np.abs(a_exact))
max_u_cd = np.max(np.abs(u_cd_out))

# CD vs analytical (CD has its own numerical error)
err_u_exact = np.max(np.abs(d_hist - u_exact)) / max_u_exact
err_v_exact = np.max(np.abs(v_hist - v_exact)) / max_v_exact

# CD vs reference CD (should be exact match)
err_u_ref = np.max(np.abs(d_hist - u_cd_out)) / max_u_cd if max_u_cd > 0 else 0
err_v_ref = np.max(np.abs(v_hist - v_cd_out)) / max_u_cd if max_u_cd > 0 else 0

tol_ref = 1e-8    # vs reference CD: essentially exact
tol_exact = 1e-1  # vs analytical: CD has inherent period error, velocity is noisier

all_pass = True

print("=" * 60)
print("  CENTRAL DIFFERENCE BENCHMARK")
print("=" * 60)
print(f"  dt/T = {dt/T:.4f},  dt/dt_crit = {dt/dt_crit:.4f}")
print(f"  u_static = {u_static*1e6:.2f} um")
print(f"  Max numerical disp = {np.max(np.abs(d_hist))*1e6:.4f} um")
print()

# vs reference CD
p1 = err_u_ref < tol_ref
p2 = err_v_ref < tol_ref
print(f"  vs SDOF CentralDifference reference:")
print(f"    Disp  rel error: {err_u_ref:.2e}  {'PASS' if p1 else 'FAIL'}")
print(f"    Vel   rel error: {err_v_ref:.2e}  {'PASS' if p2 else 'FAIL'}")
all_pass = all_pass and p1 and p2

# vs analytical (looser tolerance due to CD numerical error)
p3 = err_u_exact < tol_exact
p4 = err_v_exact < tol_exact
print(f"\n  vs Closed-Form (note: CD has inherent period error at dt/T={dt/T:.3f}):")
print(f"    Disp  rel error: {err_u_exact:.2e}  {'PASS' if p3 else 'FAIL'} (tol={tol_exact:.0e})")
print(f"    Vel   rel error: {err_v_exact:.2e}  {'PASS' if p4 else 'FAIL'} (tol={tol_exact:.0e})")
all_pass = all_pass and p3 and p4

print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
print("=" * 60 + "\n")

# ============================================================
# Plots
# ============================================================
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
fig.suptitle(
    'SDOF Truss — Central Difference vs Closed-Form\n'
    f'Step load F = {F0:.0f} N,  k = {k:.0e} N/m,  m = {m_eff:.0f} kg,  '
    f'dt/T = {dt/T:.3f}',
    fontsize=13, fontweight='bold',
)

time_ms = time_arr * 1e3

ax1.plot(time_ms, u_exact * 1e6, '-', color='black', lw=1.5, label='Exact')
ax1.plot(time_ms, u_cd_out * 1e6, '--', color='blue', lw=1.0, label='SDOF CD ref')
ax1.plot(time_ms, d_hist * 1e6, 'o', color='#e74c3c', ms=2.0, markevery=5,
         label='oneFEM (CD)')
ax1.set_ylabel(r'Displacement [$\mu$m]', fontsize=11)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.25)

ax2.plot(time_ms, v_exact * 1e3, '-', color='black', lw=1.5, label='Exact')
ax2.plot(time_ms, v_cd_out * 1e3, '--', color='blue', lw=1.0, label='SDOF CD ref')
ax2.plot(time_ms, v_hist * 1e3, 'o', color='#e74c3c', ms=2.0, markevery=5,
         label='oneFEM (CD)')
ax2.set_ylabel('Velocity [mm/s]', fontsize=11)
ax2.legend(fontsize=10, loc='upper left')
ax2.grid(True, alpha=0.25)

ax3.plot(time_ms, a_exact, '-', color='black', lw=1.5, label='Exact')
ax3.plot(time_ms, a_cd_out, '--', color='blue', lw=1.0, label='SDOF CD ref')
ax3.plot(time_ms, a_hist, 'o', color='#e74c3c', ms=2.0, markevery=5,
         label='oneFEM (CD)')
ax3.set_ylabel('Acceleration [m/s$^2$]', fontsize=11)
ax3.set_xlabel('Time [ms]', fontsize=11)
ax3.legend(fontsize=10, loc='upper left')
ax3.grid(True, alpha=0.25)

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('cd_truss_response.png', dpi=180, bbox_inches='tight')
print("Figure saved to: cd_truss_response.png")
plt.show()
