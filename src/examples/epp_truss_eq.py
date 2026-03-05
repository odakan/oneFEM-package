# Nonlinear dynamic benchmark: SDOF EPP truss under earthquake
#
# M * a_rel + C * v_rel + K_T * u_rel = -M * r * a_g(t)
#
# ElasticPerfectlyPlastic material with Newmark + Newton-Raphson.
# Reference: custom SDOF Newmark-Newton integration with EPP (pure numpy).
#
# Model: single truss, one fixed node, one free in x
#   Fixed o==========o Free
#   nd1 [0,0,0]      nd2 [L,0,0]
#

import numpy as np

# oneFEM imports
from oneFEM.model import Domain
from oneFEM.model.element.truss import Truss
from oneFEM.model.node import Node36
from oneFEM.model.element.section import Rectangular
from oneFEM.model.material.uniaxial import Elastic, ElasticPerfectlyPlastic
from oneFEM.model.tseries import Path
from oneFEM.model.pattern import UniformExcitation
from oneFEM.output.recorder import NodeRecorder, ElementRecorder
from oneFEM.analysis import Analysis
from oneFEM.analysis.algorithm import Newton
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
fy = 0.05e6      # yield stress (Pa) — low enough to trigger yielding under scaled GM

k = E * A / L
m_eff = rho_bar * L / 2.0
omega = np.sqrt(k / m_eff)
T_n = 2.0 * np.pi / omega
u_y = fy / E * L  # yield displacement

print(f"k = {k:.2e} N/m,  m = {m_eff:.0f} kg")
print(f"T = {T_n*1e3:.3f} ms,  omega = {omega:.1f} rad/s")
print(f"fy = {fy/1e6:.0f} MPa,  u_y = {u_y*1e6:.1f} um")
print()

# ============================================================
# Build oneFEM model
# ============================================================
model = Domain(nD=3)

nd1 = Node36(1, coord=[0.0, 0.0, 0.0], fix=[1, 1, 1, 1, 1, 1])
nd2 = Node36(2, coord=[L, 0.0, 0.0],   fix=[0, 1, 1, 1, 1, 1])
model.add(nd1, nd2)

mat = ElasticPerfectlyPlastic(1, E=E, fy=fy)
section = Rectangular(1, mat=mat, h=0.2, w=0.1)
tr1 = Truss(1, nodes=[nd1, nd2], section=section, rho=rho_bar, cMass=False)
model.add(tr1)

# Ground acceleration time series
gm_factor = 100.0  # scale up ground motion to trigger yielding
ts_gm = Path(1, dt=dt_record, data_values=acc_data.tolist(), factor=gm_factor)
eq_pattern = UniformExcitation(1, direction=1, tseries=ts_gm)
model.add(eq_pattern)

# Recorders
rec_disp = NodeRecorder(1, nd2, dofs=[1], results=['displacement'])
rec_vel = NodeRecorder(2, nd2, dofs=[1], results=['velocity'])
el_rec = ElementRecorder(3, tr1, results=['strain', 'stress'])
model.add(rec_disp, rec_vel, el_rec)

# Analysis: Newmark + Newton
alg = Newton(1, tangent='current', line_search=True)
const = PlainConstraints(1)
numb = PlainNumberer(1)
syst = FullGeneral(1)
integ = Newmark(1, gamma=0.5, beta=0.25)
ctest = NormUnbalance(1, tol=1e-8, maxIter=50)

analysis = Analysis(1, algorithm=alg, constraints=const,
                    integrator=integ, system=syst, test=ctest,
                    numberer=numb)

sim = SimulationManager(1, model, analysis, dt=dt_record)

print(f"Running {nSteps} steps (Newmark + Newton, EPP material) ...")
sim.analyze(nSteps, dt=dt_record)
print()

# ============================================================
# Extract results
# ============================================================
u_hist = np.array([v[0] if hasattr(v, '__getitem__') else v
                   for v in rec_disp.data['displacement']])
v_hist = np.array([v[0] if hasattr(v, '__getitem__') else v
                   for v in rec_vel.data['velocity']])
stress_hist = np.array([float(v) for v in el_rec.data['stress']])
strain_hist = np.array([float(v) for v in el_rec.data['strain']])

time_arr = np.arange(1, nSteps + 1) * dt_record

# ============================================================
# Reference: SDOF Newmark-Newton with EPP (pure numpy)
# ============================================================
gamma = 0.5
beta = 0.25
dt = dt_record

u_ref = np.zeros(nSteps + 1)
v_ref = np.zeros(nSteps + 1)
a_ref = np.zeros(nSteps + 1)
eps_p_ref = 0.0  # committed plastic strain

# Scaled acceleration for reference
acc_scaled = acc_data * gm_factor

# Initial acceleration
a_ref[0] = -acc_scaled[0]

c2 = gamma / (beta * dt)
c3 = 1.0 / (beta * dt * dt)

def epp_response(strain, eps_p_committed):
    """EPP material: returns (stress, tangent, eps_p_trial)."""
    sigma_trial = E * (strain - eps_p_committed)
    if abs(sigma_trial) <= fy:
        return sigma_trial, E, eps_p_committed
    else:
        sig = fy if sigma_trial > 0 else -fy
        return sig, 0.0, strain - sig / E

for i in range(nSteps):
    idx = min(i + 1, nSteps - 1)
    F_ext = -m_eff * acc_scaled[idx]

    # Predictor
    u_trial = u_ref[i]
    eps_p_trial = eps_p_ref

    # Newton iteration with line search
    for niter in range(50):
        strain = u_trial / L
        sigma, C_t, eps_p_trial = epp_response(strain, eps_p_ref)
        f_int = sigma * A

        # Residual
        a_trial = c3 * (u_trial - u_ref[i]) - (1.0/(beta*dt))*v_ref[i] - (1.0/(2.0*beta)-1.0)*a_ref[i]
        v_trial = v_ref[i] + dt*((1.0-gamma)*a_ref[i] + gamma*a_trial)
        R = F_ext - f_int - m_eff * a_trial

        if abs(R) < 1e-8:
            break

        # Tangent
        k_t = C_t * A / L
        k_eff = k_t + c3 * m_eff
        du = R / k_eff

        # Line search: if full step doesn't reduce |R|, bisect
        norm_old = abs(R)
        u_save = u_trial
        u_trial += du

        strain_new = u_trial / L
        sigma_new, _, _ = epp_response(strain_new, eps_p_ref)
        a_new = c3 * (u_trial - u_ref[i]) - (1.0/(beta*dt))*v_ref[i] - (1.0/(2.0*beta)-1.0)*a_ref[i]
        R_new = F_ext - sigma_new * A - m_eff * a_new

        if abs(R_new) > norm_old:
            u_trial = u_save  # undo full step
            alpha = 0.5
            for ls in range(10):
                u_ls = u_save + alpha * du
                strain_ls = u_ls / L
                sigma_ls, _, _ = epp_response(strain_ls, eps_p_ref)
                a_ls = c3 * (u_ls - u_ref[i]) - (1.0/(beta*dt))*v_ref[i] - (1.0/(2.0*beta)-1.0)*a_ref[i]
                R_ls = F_ext - sigma_ls * A - m_eff * a_ls
                if abs(R_ls) < norm_old:
                    u_trial = u_ls
                    break
                alpha *= 0.5
            else:
                u_trial = u_save + alpha * du  # accept smallest

    u_ref[i + 1] = u_trial
    a_ref[i + 1] = a_trial
    v_ref[i + 1] = v_trial
    eps_p_ref = eps_p_trial

u_ref = u_ref[1:]
v_ref = v_ref[1:]
a_ref = a_ref[1:]

# ============================================================
# Verification
# ============================================================
max_u_ref = np.max(np.abs(u_ref))
max_v_ref = np.max(np.abs(v_ref))

err_u = np.max(np.abs(u_hist - u_ref)) / max_u_ref if max_u_ref > 0 else 0
err_v = np.max(np.abs(v_hist - v_ref)) / max_v_ref if max_v_ref > 0 else 0

tol = 5e-2  # EPP + line search: path-dependent at yield transitions
all_pass = True

print("=" * 55)
print("  EPP EARTHQUAKE BENCHMARK (Newmark + Newton)")
print("=" * 55)
print(f"  Max rel. displacement: {np.max(np.abs(u_hist))*1e6:.4f} um")
print(f"  Max rel. velocity:     {np.max(np.abs(v_hist))*1e3:.4f} mm/s")
print(f"  Yield disp:            {u_y*1e6:.1f} um")

# Check if yielding occurred
n_yielded = np.sum(np.abs(stress_hist) >= fy * A * 0.999)
print(f"  Steps at yield:        {n_yielded}/{nSteps}")
print()

print(f"  vs SDOF Newmark-Newton EPP reference:")
p_u = err_u < tol
p_v = err_v < tol
print(f"    Disp  rel error: {err_u:.2e}  {'PASS' if p_u else 'FAIL'} (tol={tol:.0e})")
print(f"    Vel   rel error: {err_v:.2e}  {'PASS' if p_v else 'FAIL'} (tol={tol:.0e})")
all_pass = all_pass and p_u and p_v

print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
print("=" * 55 + "\n")

# ============================================================
# Plots
# ============================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
fig.suptitle(
    'Nonlinear Dynamic — SDOF EPP Truss under Earthquake\n'
    f'E = {E/1e9:.0f} GPa,  fy = {fy/1e6:.0f} MPa,  '
    f'T = {T_n*1e3:.3f} ms,  Newmark + Newton',
    fontsize=13, fontweight='bold',
)

# --- Displacement ---
ax = axes[0]
ax.plot(time_arr, u_ref * 1e6, '-', color='black', lw=1.2,
        label='SDOF Newmark-Newton EPP (reference)')
ax.plot(time_arr, u_hist * 1e6, '--', color='#e74c3c', lw=0.9,
        label='oneFEM', dashes=(6, 3))
ax.axhline(u_y * 1e6, color='blue', ls=':', lw=0.8, label=f'u_y = {u_y*1e6:.1f} um')
ax.axhline(-u_y * 1e6, color='blue', ls=':', lw=0.8)
ax.set_ylabel(r'$u_{rel}$ [$\mu$m]', fontsize=11)
ax.set_xlabel('Time [s]', fontsize=11)
ax.set_title('Relative Displacement', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.25)

# --- Velocity ---
ax = axes[1]
ax.plot(time_arr, v_ref * 1e3, '-', color='black', lw=1.2,
        label='Reference')
ax.plot(time_arr, v_hist * 1e3, '--', color='#e74c3c', lw=0.9,
        label='oneFEM', dashes=(6, 3))
ax.set_ylabel(r'$\dot{u}_{rel}$ [mm/s]', fontsize=11)
ax.set_xlabel('Time [s]', fontsize=11)
ax.set_title('Relative Velocity', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.25)

# --- Stress-Strain Hysteresis ---
ax = axes[2]
ax.plot(strain_hist * 1e6, stress_hist / A / 1e6, '-', color='#2c3e50', lw=0.8)
ax.axhline(fy / 1e6, color='blue', ls=':', lw=0.8, label=f'fy = {fy/1e6:.0f} MPa')
ax.axhline(-fy / 1e6, color='blue', ls=':', lw=0.8)
ax.set_xlabel(r'Strain [$\mu\epsilon$]', fontsize=11)
ax.set_ylabel('Stress [MPa]', fontsize=11)
ax.set_title('Stress-Strain Hysteresis', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('epp_truss_eq.png', dpi=180, bbox_inches='tight')
print("Figure saved to: epp_truss_eq.png")
plt.show()
