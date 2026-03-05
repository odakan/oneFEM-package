# Eigen analysis benchmark for oneFEM
#
# Test 1: SDOF truss — single element, one free translational DOF
#   Analytical: omega = sqrt(k/m), k = EA/L, m = rho*L/2 (lumped)
#
# Test 2: 2-element truss — 3 free translational DOFs
#   Compare eigenvalues against direct numpy eigh on same K_uu, M_uu
#
# Test 3: modalProperties() — verify participation factors
#
# Test 4: Triangle truss from truss.py — add mass, run eigen, show modal properties

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

all_pass = True

# ============================================================
# Test 1: SDOF truss — single element, one free DOF
# ============================================================
print("=" * 60)
print("  TEST 1: SDOF Truss Eigenvalue")
print("=" * 60)

E = 2.0e9       # Pa
A = 0.02        # m^2
L = 2.0         # m
rho_bar = 100.0 # mass per unit length (kg/m)

EA = E * A
k = EA / L
m_eff = rho_bar * L / 2.0   # lumped mass at free node
omega_exact = np.sqrt(k / m_eff)
lambda_exact = k / m_eff     # omega^2

model1 = Domain(nD=3)
nd1 = Node36(1, coord=[0.0, 0.0, 0.0], fix=[1, 1, 1, 1, 1, 1])
nd2 = Node36(2, coord=[L, 0.0, 0.0],   fix=[0, 1, 1, 1, 1, 1])
model1.add(nd1, nd2)

steel = Elastic(1, E=E)
section = Rectangular(1, mat=steel, h=0.2, w=0.1)
tr1 = Truss(1, nodes=[nd1, nd2], section=section, rho=rho_bar, cMass=False)
model1.add(tr1)

eigenvalues1 = model1.eigen(1, solver='fullGenLapack')

err1 = abs(eigenvalues1[0] - lambda_exact) / abs(lambda_exact)
pass1 = err1 < 1e-10

print(f"  k = {k:.2e} N/m")
print(f"  m_eff = {m_eff:.2e} kg (lumped)")
print(f"  omega_exact = {omega_exact:.10f} rad/s")
print(f"  lambda_exact (omega^2) = {lambda_exact:.10e}")
print(f"  lambda_oneFEM          = {eigenvalues1[0]:.10e}")
print(f"  Relative error = {err1:.2e}")
print(f"  Result: {'PASS' if pass1 else 'FAIL'}")
if not pass1:
    all_pass = False
print()


# ============================================================
# Test 2: 2-element truss — 3 free translational DOFs
# ============================================================
print("=" * 60)
print("  TEST 2: 2-Element Truss (3 free DOFs)")
print("=" * 60)

# Two elements in series: nd1(fixed) --- nd2(free x) --- nd3(free x,z)
model2 = Domain(nD=3)
nd1 = Node36(1, coord=[0.0, 0.0, 0.0], fix=[1, 1, 1, 1, 1, 1])
nd2 = Node36(2, coord=[L, 0.0, 0.0],   fix=[0, 1, 1, 1, 1, 1])
nd3 = Node36(3, coord=[L, 0.0, L],     fix=[0, 1, 0, 1, 1, 1])
model2.add(nd1, nd2, nd3)

steel2 = Elastic(1, E=E)
sec2 = Rectangular(1, mat=steel2, h=0.2, w=0.1)
tr1 = Truss(1, nodes=[nd1, nd2], section=sec2, rho=rho_bar, cMass=False)
tr2 = Truss(2, nodes=[nd2, nd3], section=sec2, rho=rho_bar, cMass=False)
model2.add(tr1, tr2)

eigenvalues2 = model2.eigen(3, solver='fullGenLapack')

# Build reference K_uu, M_uu manually and solve with numpy
# DOF layout: nd1 has DOFs 0-5 (all fixed), nd2 has DOFs 6-11 (6 free),
# nd3 has DOFs 12-17 (12,14 free)
# Free DOFs: [6, 12, 14]

# Element 1: nd1→nd2, L=2, n=[1,0,0], lumped mass rho*L/2 at each node
# Element 2: nd2→nd3, L_vert=L, n=[0,0,1], lumped mass rho*L/2 at each node
L_vert = L
K_ref = np.zeros((3, 3))
M_ref = np.zeros((3, 3))

# Element 1 contributions (DOF 6 = free x of nd2)
K_ref[0, 0] += EA / L  # k_jj for element 1, x-x

# Element 2: n=[0,0,1], nn = diag(0,0,1) in 3D → only z-z coupling
# nd2 z-DOF = 8 (fixed), nd3 z-DOF = 14 (free), nd3 x-DOF = 12 (free)
# k_jj block for element 2 (at nd3): only z-z = EA/L_vert
K_ref[2, 2] += EA / L_vert

# Mass: lumped
# nd2: m_lumped from element 1 (rho*L/2) + element 2 (rho*L_vert/2)
m_nd2_x = rho_bar * L / 2.0 + rho_bar * L_vert / 2.0  # both trusses contribute at nd2
M_ref[0, 0] = m_nd2_x
# nd3: m_lumped from element 2 only (rho*L_vert/2) on translational DOFs
m_nd3 = rho_bar * L_vert / 2.0
M_ref[1, 1] = m_nd3  # x of nd3
M_ref[2, 2] = m_nd3  # z of nd3

# Solve reference
from scipy.linalg import eigh
ref_vals, _ = eigh(K_ref, M_ref, subset_by_index=[0, 2])

print(f"  Free DOFs: 3 (nd2-x, nd3-x, nd3-z)")
print(f"  {'Mode':>4s}  {'oneFEM':>16s}  {'Reference':>16s}  {'RelErr':>12s}  {'Status':>6s}")
print("-" * 60)

test2_pass = True
for i in range(3):
    if abs(ref_vals[i]) > 0:
        err = abs(eigenvalues2[i] - ref_vals[i]) / abs(ref_vals[i])
    else:
        err = abs(eigenvalues2[i] - ref_vals[i])
    p = err < 1e-10
    if not p:
        test2_pass = False
    print(f"  {i+1:4d}  {eigenvalues2[i]:16.8e}  {ref_vals[i]:16.8e}  {err:12.2e}  {'PASS' if p else 'FAIL'}")

if not test2_pass:
    all_pass = False
print()


# ============================================================
# Test 3: modalProperties() — SDOF check
# ============================================================
print("=" * 60)
print("  TEST 3: Modal Properties (SDOF)")
print("=" * 60)

# Rebuild SDOF model (model1 already has eigen results)
props1 = model1.modalProperties()

# Verify omega
err_omega = abs(props1['omega'][0] - omega_exact) / omega_exact
pass_omega = err_omega < 1e-10
print(f"  omega: {props1['omega'][0]:.10f} vs exact {omega_exact:.10f}  "
      f"err={err_omega:.2e}  {'PASS' if pass_omega else 'FAIL'}")

# Verify frequency
f_exact = omega_exact / (2.0 * np.pi)
err_f = abs(props1['freq'][0] - f_exact) / f_exact
pass_f = err_f < 1e-10
print(f"  freq:  {props1['freq'][0]:.10f} vs exact {f_exact:.10f}  "
      f"err={err_f:.2e}  {'PASS' if pass_f else 'FAIL'}")

# Verify period
T_exact = 1.0 / f_exact
err_T = abs(props1['period'][0] - T_exact) / T_exact
pass_T = err_T < 1e-10
print(f"  T:     {props1['period'][0]:.10f} vs exact {T_exact:.10f}  "
      f"err={err_T:.2e}  {'PASS' if pass_T else 'FAIL'}")

# For SDOF with single x-DOF: effective mass = total mass = m_eff
# Gamma_x = phi^T * M * r_x, with mass-normalized phi and r_x = [1]
# phi = 1/sqrt(m_eff), so Gamma_x = 1/sqrt(m_eff) * m_eff * 1 = sqrt(m_eff)
# Meff_x = Gamma_x^2 = m_eff
err_meff = abs(props1['effectiveMass'][0, 0] - m_eff) / m_eff
pass_meff = err_meff < 1e-10
print(f"  Meff_X: {props1['effectiveMass'][0,0]:.6f} vs exact {m_eff:.6f}  "
      f"err={err_meff:.2e}  {'PASS' if pass_meff else 'FAIL'}")

# Mass participation: total mass includes fixed node mass (rho*L/2 each node)
# So total_mass_x = 2 * m_eff = 200, effective mass = m_eff = 100, ratio = 50%
expected_ratio = m_eff / (2.0 * m_eff)  # 0.5
pass_ratio = abs(props1['massParticipation'][0, 0] - expected_ratio) < 1e-10
print(f"  Mass participation X: {props1['massParticipation'][0,0]*100:.4f}% "
      f"(expected {expected_ratio*100:.1f}%)  {'PASS' if pass_ratio else 'FAIL'}")

if not (pass_omega and pass_f and pass_T and pass_meff and pass_ratio):
    all_pass = False
print()


# ============================================================
# Test 4: Triangle truss with mass — eigen + modal properties
# ============================================================
print("=" * 60)
print("  TEST 4: Triangle Truss — Modal Analysis")
print("=" * 60)

model4 = Domain(nD=3)
nd1 = Node36(1, coord=[0.0, 0.0, 0.0], fix=[1, 1, 1, 1, 1, 1])
nd2 = Node36(2, coord=[2.0, 0.0, 0.0], fix=[0, 1, 1, 1, 1, 1])
nd3 = Node36(3, coord=[2.0, 0.0, 2.0], fix=[0, 1, 0, 1, 1, 1])
model4.add(nd1, nd2, nd3)

steel4 = Elastic(1, E=E)
sec4 = Rectangular(1, mat=steel4, h=0.2, w=0.1)
tr1 = Truss(1, nodes=[nd1, nd2], section=sec4, rho=rho_bar, cMass=False)
tr2 = Truss(2, nodes=[nd2, nd3], section=sec4, rho=rho_bar, cMass=False)
tr3 = Truss(3, nodes=[nd1, nd3], section=sec4, rho=rho_bar, cMass=False)
model4.add(tr1, tr2, tr3)

eigenvalues4 = model4.eigen(3, solver='genBandArpack')
print(f"  Eigenvalues (omega^2): {eigenvalues4}")
print(f"  Frequencies (Hz): {[np.sqrt(lam)/(2*np.pi) for lam in eigenvalues4]}")

# Verify with getEigenvalue / getEigenvector
for mode in range(1, 4):
    lam = model4.getEigenvalue(mode)
    phi = model4.getEigenvector(mode)
    assert abs(lam - eigenvalues4[mode-1]) < 1e-14, f"getEigenvalue({mode}) mismatch!"
    # eigenvector should have zeros at fixed DOFs
    fixed_dofs = [0,1,2,3,4,5, 7,8,9,10,11, 13,15,16,17]  # all fixed DOFs
    for fd in fixed_dofs:
        assert abs(phi[fd]) < 1e-14, f"Eigenvector mode {mode} non-zero at fixed DOF {fd}!"
print("  getEigenvalue/getEigenvector accessor checks: PASS")

props4 = model4.modalProperties()

# Verify sum of effective masses <= total mass per direction
for d in range(3):
    total = props4['totalMass'][d]
    sum_meff = np.sum(props4['effectiveMass'][:, d])
    if total > 0:
        ratio = sum_meff / total
        print(f"  Direction {'XYZ'[d]}: total mass = {total:.2f}, sum Meff = {sum_meff:.4f}, "
              f"ratio = {ratio*100:.2f}%")
        if ratio > 1.0 + 1e-10:
            print(f"    WARNING: mass ratio > 100%!")
            all_pass = False

# Also test with fullGenLapack to compare both solvers
eigenvalues4_lapack = model4.eigen(3, solver='fullGenLapack')
err_solvers = np.max(np.abs(eigenvalues4 - eigenvalues4_lapack) /
                     np.abs(eigenvalues4_lapack))
pass_solvers = err_solvers < 1e-10
print(f"\n  Solver cross-check (ARPACK vs LAPACK): max rel err = {err_solvers:.2e}  "
      f"{'PASS' if pass_solvers else 'FAIL'}")
if not pass_solvers:
    all_pass = False

print()


# ============================================================
# Overall
# ============================================================
print("=" * 60)
print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
print("=" * 60)
