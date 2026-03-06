##-----------------------------------------------------------------------##
#  Test: All Node Specializations
#
#  Validates Node22, Node23, Node24, Node33, Node34, Node36, Node37
#  against consistent interface: construction, update, commit, revert,
#  deep-copy safety, and accessor correctness.
#
#  Pass criterion: ALL sub-tests PASS (no tolerance needed — exact match)
##-----------------------------------------------------------------------##

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from oneFEM.model.node import Node22, Node23, Node24, Node33, Node34, Node36, Node37
from oneFEM._systools.data import Vector

NODE_SPECS = [
    # (class,  nD, nDOF, description)
    (Node22,   2,  2,    "2D 2-DOF"),
    (Node23,   2,  3,    "2D 3-DOF"),
    (Node24,   2,  4,    "2D 4-DOF"),
    (Node33,   3,  3,    "3D 3-DOF"),
    (Node34,   3,  4,    "3D 4-DOF"),
    (Node36,   3,  6,    "3D 6-DOF"),
    (Node37,   3,  7,    "3D 7-DOF"),
]

def _make_coord(nD):
    return [float(i + 1) for i in range(nD)]

def _make_vec(nDOF, scale=1.0):
    return Vector([scale * (i + 1) for i in range(nDOF)], dtype=float)

def test_construction(cls, nD, nDOF, desc):
    """Test default and parameterized construction."""
    # default
    n = cls(1)
    assert n.getNDOF() == nDOF, f"nDOF mismatch"
    assert n._ID == 1, f"ID mismatch"
    # with coord
    coord = _make_coord(nD)
    n2 = cls(2, coord=coord)
    c = n2._getCoordinates()
    for i in range(nD):
        assert abs(c[i] - coord[i]) < 1e-15, f"coord[{i}] mismatch"
    # with mass
    mass = [float(i) for i in range(nDOF)]
    n3 = cls(3, mass=mass)
    # with fix
    fix = [True] * nDOF
    n4 = cls(4, fix=fix)
    f = n4._getFixity()
    for i in range(nDOF):
        assert bool(f[i]) == True, f"fix[{i}] mismatch"
    return True

def test_wrong_sizes(cls, nD, nDOF, desc):
    """Test that wrong-size inputs raise ValueError."""
    ok = True
    try:
        cls(1, coord=[1.0] * (nD + 1))
        ok = False
    except ValueError:
        pass
    try:
        cls(1, mass=[1.0] * (nDOF + 1))
        ok = False
    except ValueError:
        pass
    try:
        cls(1, fix=[True] * (nDOF + 1))
        ok = False
    except ValueError:
        pass
    return ok

def test_update_commit_revert(cls, nD, nDOF, desc):
    """Test update → commit → revert cycle with deep-copy safety."""
    n = cls(1, coord=_make_coord(nD))
    f1 = _make_vec(nDOF, 1.0)
    u1 = _make_vec(nDOF, 2.0)
    v1 = _make_vec(nDOF, 3.0)
    a1 = _make_vec(nDOF, 4.0)

    # update trial
    n._update(f1, u1, v1, a1)
    # check trial
    ut = n._getTrialDisp()
    for i in range(nDOF):
        assert abs(ut[i] - u1[i]) < 1e-15, f"trial disp[{i}]"

    # commit
    n._commitState()
    uc = n._getCommitDisp()
    for i in range(nDOF):
        assert abs(uc[i] - u1[i]) < 1e-15, f"commit disp[{i}]"

    # update with new values
    f2 = _make_vec(nDOF, 10.0)
    u2 = _make_vec(nDOF, 20.0)
    n._update(f2, u2)
    ut2 = n._getTrialDisp()
    for i in range(nDOF):
        assert abs(ut2[i] - u2[i]) < 1e-15, f"new trial disp[{i}]"

    # revert — should go back to committed state
    n._revertToLastCommit()
    ur = n._getTrialDisp()
    for i in range(nDOF):
        assert abs(ur[i] - u1[i]) < 1e-15, f"reverted disp[{i}]"

    return True

def test_deep_copy_safety(cls, nD, nDOF, desc):
    """Verify commit/revert deep-copies (no aliasing)."""
    n = cls(1, coord=_make_coord(nD))
    u1 = _make_vec(nDOF, 5.0)
    f1 = _make_vec(nDOF, 1.0)
    n._update(f1, u1)
    n._commitState()

    # mutate trial after commit
    u2 = _make_vec(nDOF, 99.0)
    n._update(f1, u2)

    # committed should be unaffected
    uc = n._getCommitDisp()
    for i in range(nDOF):
        assert abs(uc[i] - u1[i]) < 1e-15, f"commit aliased at [{i}]"

    # revert and mutate — committed should still be safe
    n._revertToLastCommit()
    u3 = _make_vec(nDOF, 77.0)
    n._update(f1, u3)
    uc2 = n._getCommitDisp()
    for i in range(nDOF):
        assert abs(uc2[i] - u1[i]) < 1e-15, f"commit aliased after revert at [{i}]"

    return True

def test_setDOF(cls, nD, nDOF, desc):
    """Test DOF index assignment."""
    n = cls(1)
    dofs = list(range(10, 10 + nDOF))
    n._setDOF(dofs)
    d = n._getDOFIndices()
    for i in range(nDOF):
        assert int(d[i]) == dofs[i], f"dof[{i}]"
    # wrong size
    try:
        n._setDOF([1])
        return False
    except ValueError:
        pass
    return True

def test_getResult(cls, nD, nDOF, desc):
    """Test result query interface."""
    n = cls(1, coord=_make_coord(nD))
    f1 = _make_vec(nDOF, 1.0)
    u1 = _make_vec(nDOF, 2.0)
    v1 = _make_vec(nDOF, 3.0)
    a1 = _make_vec(nDOF, 4.0)
    n._update(f1, u1, v1, a1)
    n._commitState()

    # query all DOFs (1-based)
    all_dofs = list(range(1, nDOF + 1))
    rd = n._getResult("disp", all_dofs)
    rv = n._getResult("vel", all_dofs)
    ra = n._getResult("accel", all_dofs)
    rf = n._getResult("force", all_dofs)
    for i in range(nDOF):
        assert abs(rd[i] - u1[i]) < 1e-15
        assert abs(rv[i] - v1[i]) < 1e-15
        assert abs(ra[i] - a1[i]) < 1e-15
        assert abs(rf[i] - f1[i]) < 1e-15

    # query subset
    rd2 = n._getResult("u", [1])
    assert abs(rd2[0] - u1[0]) < 1e-15

    # unknown query
    try:
        n._getResult("garbage", [1])
        return False
    except ValueError:
        pass
    return True

def test_revertToStart(cls, nD, nDOF, desc):
    """Test full reset."""
    n = cls(1, coord=_make_coord(nD))
    n._update(_make_vec(nDOF, 1.0), _make_vec(nDOF, 2.0))
    n._commitState()
    n._revertToStart()
    uc = n._getCommitDisp()
    for i in range(nDOF):
        assert abs(uc[i]) < 1e-15, f"not zeroed at [{i}]"
    return True

def test_repr(cls, nD, nDOF, desc):
    """Test __repr__ doesn't crash."""
    n = cls(1, coord=_make_coord(nD))
    r = repr(n)
    assert str(1) in r
    return True

# ------------------------------------------------------------------

ALL_TESTS = [
    ("construction",        test_construction),
    ("wrong_sizes",         test_wrong_sizes),
    ("update_commit_revert", test_update_commit_revert),
    ("deep_copy_safety",    test_deep_copy_safety),
    ("setDOF",              test_setDOF),
    ("getResult",           test_getResult),
    ("revertToStart",       test_revertToStart),
    ("repr",                test_repr),
]

if __name__ == "__main__":
    total = 0
    passed = 0
    failed_list = []

    for cls, nD, nDOF, desc in NODE_SPECS:
        print(f"\n--- {cls.__name__} ({desc}) ---")
        for test_name, test_fn in ALL_TESTS:
            total += 1
            try:
                ok = test_fn(cls, nD, nDOF, desc)
                if ok:
                    passed += 1
                    print(f"  {test_name}: PASS")
                else:
                    failed_list.append(f"{cls.__name__}.{test_name}")
                    print(f"  {test_name}: FAIL")
            except Exception as e:
                failed_list.append(f"{cls.__name__}.{test_name}")
                print(f"  {test_name}: FAIL ({e})")

    print(f"\n{'='*50}")
    print(f"TOTAL: {passed}/{total} PASS")
    if failed_list:
        print(f"FAILED: {', '.join(failed_list)}")
    else:
        print("ALL PASS")
