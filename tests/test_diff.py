import numpy as np

from lsq_solver.auto_diff import diff_2point, diff_3point, diff_auto


def f0(x):
    r0 = x[0] + 2 * x[1] + 3 * x[2]
    return np.array([r0])


def f1(x):
    r0 = x[0] + 2 * x[1] + 3 * x[2]
    r1 = x[1] * x[1] + x[2]
    return np.array([r0, r1])


def test_diff():
    x = np.array([1.0, 2.0, 3.0])
    j0 = diff_2point((1, 3), f0, x)
    j1 = diff_3point((1, 3), f0, x)
    j2 = diff_auto(f0, x)
    assert np.allclose(j0, j1)
    assert np.allclose(j1, j2)
