# oneFEM

A pure-Python finite element modeling and analysis package for teaching, research, and multiscale simulation.

## Features

- **Element library**: Truss (2D/3D), Elastic beam-column (2D/3D), Quad4, Hex8 (with B-bar), ZeroLength
- **Geometric nonlinearity**: Linear, P-Delta, and Corotational coordinate transformations (2D/3D)
- **Continuum kinematics**: Linear (infinitesimal strain), Total Lagrangian, Updated Lagrangian
- **Materials**: Elastic, ElasticPerfectlyPlastic (uniaxial); ElasticIsotropic (PlaneStress/PlaneStrain/3D)
- **Analysis**: Static (Linear, Newton-Raphson), Dynamic (Newmark, Central Difference), Eigenvalue
- **Integrators**: LoadControl, DisplacementControl, Newmark, CentralDifference
- **Solvers**: Dense (numpy), Sparse (UMFPACK via scipy)
- **Recorders**: Node, Element, and ModeShape recorders with file output

## Installation

```bash
# Development install
pip install -e .

# Or install dependencies only
pip install numpy scipy matplotlib ipython
```

### Optional: UMFPACK sparse solver (Linux)

```bash
sudo apt install ninja-build swig libopenblas-dev liblapack-dev libsuitesparse-dev
pip install numpy==1.26.4
pip install --force-reinstall --no-deps scipy scikit-umfpack
```

## Quick Start

```python
from oneFEM.model import Domain
from oneFEM.model.node import Node36
from oneFEM.model.element.truss import Truss
from oneFEM.model.material.uniaxial import Elastic
from oneFEM.model.pattern import Plain
from oneFEM.model.tseries import Constant
from oneFEM.analysis import Analysis
from oneFEM import SimulationManager
from oneFEM.output.recorder import NodeRecorder

# Build model
model = Domain()
model.add(Node36(1, coord=[0, 0, 0], fix=[True]*6))
model.add(Node36(2, coord=[1, 0, 0]))

mat = Elastic(1, E=200e9)
model.add(Truss(1, [1, 2], mat, A=0.01))

ts = Constant(1, factor=1.0)
pat = Plain(1, ts, loads=[[2, 1000, 0, 0, 0, 0, 0]])
model.add(pat)

rec = NodeRecorder(1, nd=2, dofs=[1, 2, 3], results=['disp'])
model.add(rec)

# Analyze
analysis = Analysis(algorithm='Linear', system='FullGeneral',
                    integrator='LoadControl', numberer='Plain',
                    constraints='Plain')
sim = SimulationManager(model, analysis)
sim.analyze(nSteps=1)
```

## Benchmarks

All benchmarks run from `src/`:

| Benchmark | Script | Tests | Status |
|-----------|--------|-------|--------|
| 3D Truss (static) | `examples/truss.py` | 3 | ALL PASS |
| Truss (dynamic, Newmark) | `dynamic_truss.py` | 3 | ALL PASS |
| Eigenvalue/modal | `eigen_truss.py` | 4+ | ALL PASS |
| Nonlinear static (EPP) | `epp_truss.py` | 3 | ALL PASS |
| Beam 2D/3D | `examples/beam.py` | 10 | ALL PASS |
| Column buckling (PDelta/Corot) | `examples/column_buckling.py` | 8 | ALL PASS |
| Corot beam (snap-through, Lee's frame) | `examples/corot_benchmarks.py` | 5 | ALL PASS |
| Quad4 patch test | `examples/quad4_patch_test.py` | 6 | ALL PASS |
| Quad4 Cook's membrane | `examples/quad4_cooks_membrane.py` | 2 | ALL PASS |
| Hex8 B-bar (patch, cylinder, cantilever, Cook's 3D) | `examples/hex8_benchmarks.py` | 16 | ALL PASS |

Run all benchmarks:

```bash
cd src
python examples/truss.py
python examples/beam.py
python examples/column_buckling.py
python examples/corot_benchmarks.py
python examples/quad4_patch_test.py
python examples/quad4_cooks_membrane.py
python examples/hex8_benchmarks.py
```

## Package Structure

```
src/oneFEM/
├── _systools/          # Internal utilities (Vector, Matrix, Tensor, CTensor)
├── model/              # FEM model definition
│   ├── node/           # Node specializations (Node22, Node23, Node33, Node36, ...)
│   ├── element/        # Elements (truss, beam, continuum, shell, zerolength)
│   │   ├── continuum/  # Quad4, Hex8 (with B-bar)
│   │   └── kinematics/ # Linear, TL, UL, CrdTransf (Linear/PDelta/Corot)
│   ├── material/       # Uniaxial (Elastic, EPP) and nD (ElasticIsotropic)
│   ├── pattern/        # Load patterns with time series
│   └── constraint/     # Boundary conditions
├── analysis/           # Solver pipeline
│   ├── algorithm/      # Linear, Newton-Raphson, Krylov-Newton
│   ├── integrator/     # Static and dynamic integrators
│   ├── system/         # Equation solvers
│   └── eigen/          # Eigenvalue solvers
└── output/             # Recorders (Node, Element, ModeShape)
```

## Requirements

- Python >= 3.7
- NumPy >= 1.21
- SciPy >= 1.7
- Matplotlib >= 3.4

## License

GPLv3 — see [LICENSE](LICENSE) for details.
