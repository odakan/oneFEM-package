# Architecture Decision Records (ADR)
# oneFEM — Decisions that are CLOSED. Do not re-argue without opening a new ADR.
#
# Format: ADR-NNN: Title
#   Status: ACCEPTED | SUPERSEDED | DEPRECATED
#   Date: YYYY-MM
#   Context: why the decision arose
#   Decision: what was decided
#   Consequences: what this means for future code
#   Rejected alternatives: what was explicitly ruled out
# ─────────────────────────────────────────────────────────────────────────────

## ADR-001: Duck Typing Over ABC / Protocol for Material and Element Interfaces

**Status:** ACCEPTED
**Date:** 2024

**Context:**
Python offers three ways to define interfaces: abstract base classes (ABC),
`typing.Protocol`, and implicit duck typing. All three were considered for
the Material and Element interfaces.

**Decision:**
Use implicit duck typing. No ABCs, no Protocol enforcement in production code.
`typing.Protocol` stubs are planned as documentation-only in `model/protocols.py`
(see Section 18 of blueprint) but are not enforced at runtime.

**Consequences:**
- No inheritance required for `NeuralMaterialSurrogate` or `MicroSimulationManager`
  to act as a Material. They just need the right methods.
- Errors surface at call time (AttributeError) not at class definition time.
- `isinstance(obj, UniaxialMaterial)` will not work until `model/protocols.py`
  is added with `@runtime_checkable`.

**Rejected alternatives:**
- ABC with `@abstractmethod`: forces inheritance, makes neural surrogate injection
  awkward (surrogate has a completely different implementation heritage)
- Full Protocol enforcement now: premature — no type hints in codebase yet,
  adding Protocol enforcement without type hints creates inconsistency


## ADR-002: No Type Hints in Production Code

**Status:** ACCEPTED
**Date:** 2024

**Context:**
Type hints improve IDE support and catch interface errors early. However,
the existing codebase has zero type hints and is already ~5K lines.

**Decision:**
No type hints added to existing code. New code follows the same convention.
A future ADR may reverse this if a decision is made to migrate the whole
codebase at once with a dedicated sprint.

**Consequences:**
- Do not add `def _setTrialStrain(self, strain: float) -> int:` in new code.
- Do not add `-> None`, `-> np.ndarray`, etc.
- Interface documentation lives in docstrings and the blueprint, not in signatures.

**Rejected alternatives:**
- Partial adoption (new files only): creates inconsistency that confuses both
  humans and LLMs — half the codebase typed, half not


## ADR-003: Private Attribute Naming — Double vs Single Underscore

**Status:** ACCEPTED
**Date:** 2024

**Context:**
Python name mangling (`__attr`) prevents accidental access from subclasses.
Double underscore in `Domain` prevents external code from directly touching
`__node_map` etc. But name mangling in Node subclasses caused bugs: the base
class couldn't access attributes defined with `__` in subclasses.

**Decision:**
- `self.__attr` (double underscore): **Domain internals only**
- `self._attr` (single underscore): **all other classes including Node subclasses**

This is a hard rule, not a style preference. Violating it in Node subclasses
causes silent AttributeError when the base class calls `getNDOF()`, `getDOFs()`,
`_fix`, etc.

**Consequences:**
- Every new Node specialization uses `_coord`, `_dofs`, `_fix`, `_u_trial`, etc.
- Every new Material uses `_eps_trial`, `_sig_trial`, etc.
- Only code in `model/main.py` (Domain) may use `__`.

**Rejected alternatives:**
- Double underscore everywhere: broke base class access in Node hierarchy
- Single underscore everywhere: `Domain.__node_map` loses its encapsulation,
  external code can accidentally touch it


## ADR-004: Deep Copy in Commit/Revert — Vector Constructor, Not Assignment

**Status:** ACCEPTED
**Date:** 2024

**Context:**
Early implementations used `self._u_commit = self._u_trial` for the commit
operation. This creates an alias — both names point to the same numpy array.
When `_u_trial` is subsequently modified during Newton iteration, `_u_commit`
changes too, silently destroying the committed state. Revert then restores the
*modified* (wrong) state, not the last converged state. This bug produces
wrong results with no error or warning.

**Decision:**
All commit and revert operations use the `Vector` constructor for deep copy:

```python
# CORRECT:
self._u_commit = Vector(self._u_trial)   # new Vector from data copy

# WRONG — do not use:
self._u_commit = self._u_trial           # alias — same array object
self._u_commit = self._u_trial.copy()   # bypasses Vector wrapper
self._u_commit[:] = self._u_trial       # ok for numpy but not Vector
```

For scalar state (float), Python scalars are immutable so aliasing is safe,
but we use explicit assignment anyway for consistency and clarity.

**Consequences:**
- Every `_commitState()` and `_revertToLastCommit()` in Node, Material, and
  Element must use `Vector(...)` for all state vectors.
- This rule applies retroactively — if you find aliasing in existing code,
  fix it immediately and add a regression test.

**Rejected alternatives:**
- In-place copy `[:] =`: works for numpy arrays but doesn't preserve Vector
  wrapper semantics; also requires pre-allocation


## ADR-005: Pipeline Order — Domain Commit Before Element Update

**Status:** ACCEPTED
**Date:** 2024

**Context:**
After Newton convergence, the sequence is:
1. Domain updates node committed state from the converged displacement vector
2. Elements read the updated nodal displacements and update their internal state
3. Recorders observe the committed state

The question was whether elements should be updated before or after nodes commit.

**Decision:**
`Domain._commit()` (node state update) **must** run before `element._update()`
and `element._commit()`. The fixed sequence is:

```
Domain._commit(F, u)        # 1. nodes commit
  └── element._update()     # 2. elements read updated nodal state
  └── element._commit()     # 3. elements commit their materials
Domain._record(time)        # 4. recorders observe committed state
```

**Consequences:**
- Elements always read current-step (not previous-step) nodal displacements.
- Recorders always see fully committed state — never trial state.
- Do not reorder these calls. Do not call `element._update()` before
  `Domain._commit()` for "efficiency" reasons.

**Rejected alternatives:**
- Elements update first: elements would compute strain from previous-step
  displacements, one step behind. Produces silently wrong stress output.


## ADR-006: SoC Unified Memory as the Primary Hardware Target (WP23-24)

**Status:** ACCEPTED
**Date:** 2024

**Context:**
GPU-accelerated FEM typically uses discrete GPUs with explicit host↔device
copies. At moderate problem size (50K–500K DOF), copy overhead on PCIe
dominates solve time. Two approaches were considered:
1. Optimize copies (async transfer, double buffering)
2. Eliminate copies structurally via unified memory SoC

**Decision:**
Target unified memory SoC (NVIDIA GH200, AMD MI300A) as the primary hardware
platform. The `UnifiedAllocator` (WP23-24) allocates into the unified address
space — CPU assembly and GPU solve operate on the same physical memory.

The Tier-1 acceptance gate is binary: **the analysis must complete with zero
explicit `cudaMemcpy` or `hipMemcpy` calls on GH200/MI300A hardware.**

On discrete GPUs, `cupy.asarray()` still performs a PCIe copy — this is
acceptable as a fallback, not as the target.

**Consequences:**
- `SparseMatrixSOE` assembly (CPU) and solve (GPU) must share a single
  allocation — no intermediate CPU-side numpy arrays that get copied.
- The Python benchmark (`benchmark_unified_memory.py`) provides hardware-portable
  evidence of the architectural gain. It is a grant deliverable — do not modify
  its methodology without PI approval (see Stop and Ask Rules).
- Validation device for WP23-24 Python layer: Jetson AGX Orin ($499)
- Validation device for WP23-24 C++ layer: EuroHPC GH200 nodes

**Rejected alternatives:**
- Async copy optimization: reduces but does not eliminate copy overhead;
  does not support the strong scientific claim ("zero copies")
- Keeping discrete GPU as primary target: scientifically weaker claim,
  loses differentiation from existing GPU-FEM literature


## ADR-008: Python With Hardware-Native Backends Is the Primary Deliverable

**Status:** ACCEPTED
**Date:** 2024

**Context:**
Early documentation framed oneFEM as a "Python mockup" for a future C++ port,
with WP10 (MPI), WP11 (TBB), and WP13 (CUDA) labelled "C++ only." This created
the wrong mental model: contributors assumed Python was a throwaway prototype
and that real performance would come from a separate C++ implementation.

**Decision:**
oneFEM is a pure Python research platform. It is not a prototype for a C++ port.
Performance is delivered entirely through hardware-native Python backends:

- GPU sparse solve: `cupyx.scipy.sparse.linalg` (cuSPARSE backend)
- Unified memory: `cupy.cuda.UnifiedMemory` (cudaMallocManaged)
- Distributed: `mpi4py` (MPI backend)
- Shared parallel: `concurrent.futures.ThreadPoolExecutor`
- Neural surrogate: `torch` / `jax` with autograd consistent tangent

C++ appears in exactly one role: OpenSees is the reference implementation
whose interfaces we mirror (Section 17). We do not plan to rewrite oneFEM in C++.

**Consequences:**
- Every WP has a Python implementation target with a named Python backend library.
- When a performance tool is needed, reach for a Python backend first.
  Only propose a C++ extension if a Python backend genuinely cannot meet
  the requirement — and document why in a new ADR.
- The scientific contribution is the Python abstraction itself: a
  unified-memory, hardware-portable, research-grade FEM platform built
  entirely in Python.
- The benchmark in `benchmark_unified_memory.py` compares Python-vs-Python
  across hardware platforms. This is a clean comparison because Python
  overhead is identical on both sides — the only variable is hardware architecture.

**Rejected alternatives:**
- C++ port as primary target: not planned; would require a separate team and
  timeline; the Python backends are sufficient for the scientific claims
- Cython/pybind11 extensions for performance: not needed; CuPy and mpi4py
  already provide C-level performance for the bottleneck operations


**Status:** ACCEPTED
**Date:** 2024

**Context:**
Python conventions favour `@property` for attribute access. OpenSees uses
Java-style `getX()` / `setX()` methods. The question was which to adopt.

**Decision:**
Use `getX()` / `setX()` style (OpenSees convention) for all FEM domain
objects (Node, Element, Material, Domain). Use `@property` only for the
`Vector` / `Matrix` data wrappers in `_systools/data/` (`.data`, `.length`,
`.size`).

**Consequences:**
- `node.getDisp()` not `node.disp`
- `material.getStress()` not `material.stress`
- Consistent with OpenSees source — makes porting from C++ to Python
  and reading OpenSees docs unambiguous.
- `@property` is acceptable for new utility/infrastructure classes outside
  the FEM domain layer.

**Rejected alternatives:**
- `@property` everywhere: would diverge from OpenSees naming and make the
  Section 17 mapping table misleading
