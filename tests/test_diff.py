from itertools import product

import numpy as np
import pytest

from lsq_solver.auto_diff import AUTO_DIFF_NAMES, make_jac


def f_one_input_one_output(x):
    r0 = x[0] + 2 * x[1] + 3 * x[2] * x[2]
    return np.array([r0])


def f_one_input_one_output_jac(x):
    return np.array([1.0, 2.0, 6.0 * x[2]])


def f_one_input_two_output(x):
    r0 = x[0] + 2 * x[1] + 3 * x[2]
    r1 = x[1] * x[1] + x[2]
    return np.array([r0, r1])


def f_one_input_two_output_jac(x):
    return np.array([[1, 2, 3.0], [0.0, 2 * x[1], 1]])


FUNC_JAC_PAIR_LIST = [
    (f_one_input_one_output, f_one_input_one_output_jac, (1, 3)),
    (f_one_input_two_output, f_one_input_two_output_jac, (2, 3)),
]


@pytest.mark.parametrize("auto_diff_name, func_and_jac_gt", product(AUTO_DIFF_NAMES, FUNC_JAC_PAIR_LIST))
def test_diff(auto_diff_name, func_and_jac_gt):
    func, jac_gt, shape = func_and_jac_gt
    x = np.array([2.0, 3.0, 4.0])
    jac = make_jac(auto_diff_name, shape, func)(x)
    assert np.allclose(jac, jac_gt(x))
