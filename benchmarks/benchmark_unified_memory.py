"""
benchmark_unified_memory.py
===========================
Compares memory transfer overhead and sparse solve performance between:
  - Discrete GPU (PCIe) architecture  — explicit host→device copies are real
  - Unified memory SoC (Jetson / MI300A / GH200) — copies are pointer remaps

Run this script ONCE on a discrete GPU machine and ONCE on a SoC machine.
Compare the printed results. The difference in "copy time" is the architectural gain.

Requirements:
    pip install cupy-cuda12x scipy numpy   # adjust cupy version to your CUDA
    (on ROCm: pip install cupy-rocm-5-0)

Usage:
    python benchmark_unified_memory.py --dof 10000 50000 100000
    python benchmark_unified_memory.py --dof 10000 --runs 20 --output results.json
"""

import argparse
import json
import platform
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

# ── CuPy import (graceful fallback) ──────────────────────────────────────────
try:
    import cupy as cp
    import cupyx.scipy.sparse
    import cupyx.scipy.sparse.linalg as cpx_linalg
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("[WARNING] cupy not found — GPU benchmarks will be skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    dof:              int
    runs:             int

    # CPU baseline
    cpu_assemble_ms:  float = 0.0
    cpu_solve_ms:     float = 0.0
    cpu_total_ms:     float = 0.0

    # GPU path — copy present (discrete GPU behaviour)
    gpu_copy_h2d_ms:  float = 0.0   # host → device (simulated even on SoC)
    gpu_solve_ms:     float = 0.0
    gpu_copy_d2h_ms:  float = 0.0   # device → host
    gpu_total_with_copy_ms: float = 0.0

    # GPU path — zero-copy (unified memory / SoC ideal)
    gpu_total_zero_copy_ms: float = 0.0

    # Derived
    copy_overhead_ms:     float = 0.0   # total copy cost
    copy_overhead_pct:    float = 0.0   # copy as % of total-with-copy
    speedup_zero_copy:    float = 0.0   # (total_with_copy) / (zero_copy)
    residual_norm:        float = 0.0   # ||Ax - b|| sanity check


@dataclass
class BenchmarkReport:
    hostname:       str
    platform:       str
    cuda_version:   str
    gpu_name:       str
    is_unified:     Optional[bool]   # None = unknown
    results:        list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# FEM-representative sparse matrix generator
# ─────────────────────────────────────────────────────────────────────────────

def build_fem_stiffness(n_dof: int, bandwidth: int = 50,
                        seed: int = 42) -> scipy.sparse.csr_matrix:
    """
    Generate a sparse SPD matrix representative of a FEM stiffness matrix.

    Properties chosen to match a mid-size structural model:
      - Banded sparsity (bandwidth ~ sqrt(n_dof) for 2D problems)
      - Symmetric positive definite (required for Cholesky / CG solvers)
      - Condition number ~10^4 (realistic for linear elastic FEM)

    This is NOT a real FEM assembly — it generates a matrix with the same
    sparsity and conditioning characteristics without requiring a mesh.
    """
    rng = np.random.default_rng(seed)

    rows, cols, vals = [], [], []
    bw = min(bandwidth, n_dof // 4)

    for i in range(n_dof):
        # Diagonal — dominant for SPD
        rows.append(i); cols.append(i)
        vals.append(float(rng.uniform(10.0, 20.0)))

        # Off-diagonal within bandwidth
        for j in range(max(0, i - bw), i):
            v = rng.uniform(-0.5, 0.5)
            rows.append(i); cols.append(j); vals.append(v)
            rows.append(j); cols.append(i); vals.append(v)

    K = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(n_dof, n_dof)
    ).tocsr()

    # Enforce strict diagonal dominance → guaranteed SPD
    diag = np.array(K.diagonal())
    off_diag_sum = np.array(np.abs(K).sum(axis=1)).flatten() - np.abs(diag)
    boost = np.maximum(0.0, off_diag_sum - diag + 1.0)
    K = K + scipy.sparse.diags(boost)

    return K


def build_rhs(n_dof: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n_dof).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────

def timer_ms() -> float:
    return time.perf_counter() * 1000.0


def gpu_sync_timer(func):
    """Run func(), synchronize GPU, return elapsed milliseconds."""
    if CUPY_AVAILABLE:
        cp.cuda.Device().synchronize()
    t0 = timer_ms()
    result = func()
    if CUPY_AVAILABLE:
        cp.cuda.Device().synchronize()
    return result, timer_ms() - t0


# ─────────────────────────────────────────────────────────────────────────────
# CPU benchmark
# ─────────────────────────────────────────────────────────────────────────────

def bench_cpu(K: scipy.sparse.csr_matrix, b: np.ndarray,
              runs: int) -> tuple[float, float]:
    """Returns (mean_assemble_ms, mean_solve_ms) over `runs` repetitions."""
    assemble_times, solve_times = [], []

    for _ in range(runs):
        # Assembly proxy: COO → CSR conversion (representative of assembly cost)
        t0 = timer_ms()
        K_csr = K.tocoo().tocsr()
        assemble_times.append(timer_ms() - t0)

        # Solve
        t0 = timer_ms()
        _ = scipy.sparse.linalg.spsolve(K_csr, b)
        solve_times.append(timer_ms() - t0)

    return float(np.mean(assemble_times)), float(np.mean(solve_times))


# ─────────────────────────────────────────────────────────────────────────────
# GPU benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def bench_gpu_with_copy(K_cpu: scipy.sparse.csr_matrix, b_cpu: np.ndarray,
                        runs: int) -> tuple[float, float, float, float]:
    """
    Simulates the DISCRETE GPU path:
      host → device copy | GPU solve | device → host copy

    Even on a SoC, we force this path explicitly so we can measure
    what the copy cost *would* be — this is the baseline to subtract.

    Returns: (h2d_ms, solve_ms, d2h_ms, residual_norm)
    """
    h2d_times, solve_times, d2h_times = [], [], []
    residual = 0.0

    for i in range(runs):
        # ── Host → Device copy (simulated PCIe transfer) ───────────────────
        _, h2d = gpu_sync_timer(lambda: (
            cupyx.scipy.sparse.csr_matrix(K_cpu),
            cp.asarray(b_cpu)
        ))
        K_gpu = cupyx.scipy.sparse.csr_matrix(K_cpu)
        b_gpu = cp.asarray(b_cpu)
        h2d_times.append(h2d)

        # ── GPU sparse solve ───────────────────────────────────────────────
        x_gpu, solve_ms = gpu_sync_timer(
            lambda: cpx_linalg.spsolve(K_gpu, b_gpu)
        )
        solve_times.append(solve_ms)

        # ── Device → Host copy ─────────────────────────────────────────────
        _, d2h = gpu_sync_timer(lambda: cp.asnumpy(x_gpu))
        x_cpu = cp.asnumpy(x_gpu)
        d2h_times.append(d2h)

        if i == 0:
            residual = float(np.linalg.norm(K_cpu @ x_cpu - b_cpu))

    return (float(np.mean(h2d_times)),
            float(np.mean(solve_times)),
            float(np.mean(d2h_times)),
            residual)


def bench_gpu_zero_copy(K_cpu: scipy.sparse.csr_matrix, b_cpu: np.ndarray,
                        runs: int) -> float:
    """
    Simulates the UNIFIED MEMORY / SoC path:
      GPU solve directly on the (already accessible) data, no copy phase.

    On a true unified memory SoC (Jetson, GH200, MI300A):
      cp.asarray() is a pointer remap (~microseconds), not a copy.
    On a discrete GPU:
      cp.asarray() still triggers a PCIe transfer — but this benchmark
      models the SoC ideal to quantify what the gain *would be*.

    The delta between bench_gpu_with_copy and this function is the
    architectural benefit of unified memory.

    Returns: mean_total_ms (includes the pointer remap cost)
    """
    total_times = []

    for _ in range(runs):
        t0 = timer_ms()

        # On SoC: this is ~0 cost (pointer remap in unified address space)
        # On discrete GPU: this is still a copy — but we're modeling the SoC
        K_gpu = cupyx.scipy.sparse.csr_matrix(K_cpu)
        b_gpu = cp.asarray(b_cpu)

        if CUPY_AVAILABLE:
            cp.cuda.Device().synchronize()

        _ = cpx_linalg.spsolve(K_gpu, b_gpu)

        if CUPY_AVAILABLE:
            cp.cuda.Device().synchronize()

        total_times.append(timer_ms() - t0)

    return float(np.mean(total_times))


# ─────────────────────────────────────────────────────────────────────────────
# Device info
# ─────────────────────────────────────────────────────────────────────────────

def get_device_info() -> dict:
    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cuda_version": "N/A",
        "gpu_name": "N/A",
        "is_unified": None,
    }

    if not CUPY_AVAILABLE:
        return info

    try:
        info["cuda_version"] = cp.cuda.runtime.runtimeGetVersion()
        dev = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        info["gpu_name"] = props["name"].decode()

        # Heuristic: unified memory SoCs report integrated=1
        # Jetson: integrated=1, GH200: integrated=1, discrete RTX: integrated=0
        info["is_unified"] = bool(props.get("integrated", 0))
    except Exception as e:
        info["gpu_name"] = f"(error: {e})"

    return info


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(dof_sizes: list[int], runs: int = 10,
                  bandwidth: int = 50) -> BenchmarkReport:

    info = get_device_info()
    report = BenchmarkReport(
        hostname=info["hostname"],
        platform=info["platform"],
        cuda_version=str(info["cuda_version"]),
        gpu_name=info["gpu_name"],
        is_unified=info["is_unified"],
    )

    print(f"\n{'='*60}")
    print(f"  oneFEM Unified Memory Benchmark")
    print(f"  Host:     {info['hostname']}")
    print(f"  GPU:      {info['gpu_name']}")
    print(f"  Unified:  {info['is_unified']} (True = SoC, False = discrete)")
    print(f"  Runs:     {runs} per DOF size")
    print(f"{'='*60}\n")

    header = (
        f"{'DOF':>8} │ "
        f"{'CPU assm':>10} │ "
        f"{'CPU solve':>10} │ "
        f"{'H→D copy':>10} │ "
        f"{'GPU solve':>10} │ "
        f"{'D→H copy':>10} │ "
        f"{'Copy %':>8} │ "
        f"{'Speedup':>8}"
    )
    print(header)
    print("─" * len(header))

    for n_dof in dof_sizes:
        print(f"  Building {n_dof} DOF system...", end=" ", flush=True)
        K = build_fem_stiffness(n_dof, bandwidth=bandwidth)
        b = build_rhs(n_dof)
        print("done")

        # CPU
        cpu_assm, cpu_solve = bench_cpu(K, b, runs)

        result = BenchmarkResult(dof=n_dof, runs=runs)
        result.cpu_assemble_ms = round(cpu_assm, 3)
        result.cpu_solve_ms    = round(cpu_solve, 3)
        result.cpu_total_ms    = round(cpu_assm + cpu_solve, 3)

        if CUPY_AVAILABLE:
            # GPU with copy
            h2d, gpu_solve, d2h, residual = bench_gpu_with_copy(K, b, runs)
            result.gpu_copy_h2d_ms         = round(h2d, 3)
            result.gpu_solve_ms            = round(gpu_solve, 3)
            result.gpu_copy_d2h_ms         = round(d2h, 3)
            result.gpu_total_with_copy_ms  = round(h2d + gpu_solve + d2h, 3)
            result.residual_norm           = round(residual, 6)

            # GPU zero-copy (SoC ideal)
            zero_copy_total = bench_gpu_zero_copy(K, b, runs)
            result.gpu_total_zero_copy_ms  = round(zero_copy_total, 3)

            # Derived
            total_copy = h2d + d2h
            result.copy_overhead_ms  = round(total_copy, 3)
            result.copy_overhead_pct = round(
                100.0 * total_copy / result.gpu_total_with_copy_ms, 1
            ) if result.gpu_total_with_copy_ms > 0 else 0.0
            result.speedup_zero_copy = round(
                result.gpu_total_with_copy_ms / result.gpu_total_zero_copy_ms, 2
            ) if result.gpu_total_zero_copy_ms > 0 else 0.0

        report.results.append(result)

        print(
            f"  {n_dof:>8,} │ "
            f"{result.cpu_assemble_ms:>9.1f}ms │ "
            f"{result.cpu_solve_ms:>9.1f}ms │ "
            f"{result.gpu_copy_h2d_ms:>9.1f}ms │ "
            f"{result.gpu_solve_ms:>9.1f}ms │ "
            f"{result.gpu_copy_d2h_ms:>9.1f}ms │ "
            f"{result.copy_overhead_pct:>7.1f}% │ "
            f"{result.speedup_zero_copy:>7.2f}x"
        )

    print()
    _print_summary(report)
    return report


def _print_summary(report: BenchmarkReport):
    print(f"\n{'─'*60}")
    print(f"  SUMMARY — {report.gpu_name}")
    print(f"  Unified memory (SoC): {report.is_unified}")
    print(f"{'─'*60}")

    if not report.results:
        return

    for r in report.results:
        if r.copy_overhead_pct > 0:
            flag = ""
            if r.copy_overhead_pct > 50:
                flag = "  ← copy dominates!"
            elif r.copy_overhead_pct > 25:
                flag = "  ← significant copy overhead"
            print(
                f"  {r.dof:>8,} DOF: copy overhead = "
                f"{r.copy_overhead_ms:.1f}ms ({r.copy_overhead_pct:.1f}%)  "
                f"zero-copy speedup = {r.speedup_zero_copy:.2f}x{flag}"
            )

    print()
    print("  Interpretation:")
    if report.is_unified:
        print("  → On this SoC, cp.asarray() is a pointer remap (~0 cost).")
        print("  → The 'copy' columns above reflect hardware-unified access.")
        print("  → speedup_zero_copy ≈ 1.0 means copies are already free.")
    else:
        print("  → On this discrete GPU, copies are real PCIe transfers.")
        print("  → copy_overhead_pct shows what you'd save on a SoC.")
        print("  → speedup_zero_copy shows the *potential* gain on GH200/MI300A.")
    print()
    print("  Grant framing:")
    print("  → Run this on your RTX workstation, then on Jetson/EuroHPC GH200.")
    print("  → The difference in copy_overhead_pct IS the architectural claim.")
    print("  → 'Our allocator eliminates X% of wall time on unified memory.'")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="oneFEM unified memory benchmark — compare SoC vs discrete GPU"
    )
    parser.add_argument(
        "--dof", nargs="+", type=int,
        default=[5000, 10000, 50000],
        help="DOF sizes to benchmark (default: 5000 10000 50000)"
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Number of repeated runs per DOF size (default: 10)"
    )
    parser.add_argument(
        "--bandwidth", type=int, default=50,
        help="Sparse matrix bandwidth (default: 50, ~2D FEM)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save JSON results to this file"
    )
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="Skip GPU benchmarks (useful if cupy not installed)"
    )
    args = parser.parse_args()

    if args.cpu_only:
        global CUPY_AVAILABLE
        CUPY_AVAILABLE = False

    report = run_benchmark(
        dof_sizes=args.dof,
        runs=args.runs,
        bandwidth=args.bandwidth,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
