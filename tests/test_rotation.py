from lsq_solver.rotation import rotation_matrix
import numpy as np
from scipy.spatial.transform import Rotation

def test_rotation():
    rvec = np.array([0.1, 0.2, 0.3])
    rmat_scipy = Rotation.from_rotvec(rvec).as_matrix()
    rmat = rotation_matrix(rvec)
    assert np.allclose(rmat_scipy, rmat)