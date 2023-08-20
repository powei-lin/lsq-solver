from functools import partial

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lsq_solver import LeastSquaresProblem
from lsq_solver.auto_diff import AUTO_DIFF_NAMES
from lsq_solver.rotation import rotation_matrix


def cost(p3ds: np.ndarray, p2ds: np.ndarray, rvec: np.ndarray) -> np.ndarray:
    p3dd = rotation_matrix(rvec) @ p3ds
    return (p3dd[:2, :] / p3dd[2:, :] - p2ds).flatten()


def test_rotation():
    rvec = np.array([0.1, 0.2, 0.3])
    rmat_scipy = Rotation.from_rotvec(rvec).as_matrix()
    rmat = rotation_matrix(rvec)
    assert np.allclose(rmat_scipy, rmat)


@pytest.mark.parametrize("auto_diff_name", AUTO_DIFF_NAMES)
def test_rotation_problem(auto_diff_name: str):
    np.random.seed(0)
    p3ds_gt = (np.random.random((100, 3)) * 100 + np.array([0, 0, 1])).T
    rvec_gt = np.random.random(3)
    rmat_gt = Rotation.from_rotvec(rvec_gt).as_matrix()
    p3ds_gt_r = rmat_gt @ p3ds_gt
    p2ds_gt = p3ds_gt_r[:2, :] / p3ds_gt_r[2:, :]

    rvec = rvec_gt + np.random.random(3) / 100
    rmat = Rotation.from_rotvec(rvec).as_matrix()
    p3ds_r = rmat @ p3ds_gt
    problem = LeastSquaresProblem()
    cc = partial(cost, p3ds_gt, p2ds_gt)
    problem.add_residual_block(p3ds_r.shape[1] * 2, cc, rvec, jac_func=auto_diff_name)
    problem.solve()
    assert np.allclose(rvec_gt, rvec)
