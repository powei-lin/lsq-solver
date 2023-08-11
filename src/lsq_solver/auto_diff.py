from functools import partial
import numpy as np
from num_dual import jacobian, gradient
from typing import Callable
# the value follows the wiki https://en.wikipedia.org/wiki/Numerical_differentiation
ROUNDING_ERROR = 1.48e-8


def make_jac(name: str, jac_shape, func: callable):
    if name == '2-point':
        return partial(diff_2point, jac_shape, func)    # partial other parameters since we need a single-input function
    elif name == '3-point':
        return partial(diff_3point, jac_shape, func)
    elif name == 'dual':
        return partial(diff_auto, func)
    else:
        raise ValueError('Unsupported jacobian function.')


def diff_2point(jac_shape, func: callable, *variables):
    """
    2-point numeric finite difference. f'(x) = (f(x+h) - f(x)) / h
    """
    f0 = func(*variables)
    jac = np.zeros(jac_shape, dtype=np.float64)

    jac_col = 0
    for variable in variables:
        h = np.maximum(ROUNDING_ERROR * variable, ROUNDING_ERROR)
        for j in range(h.shape[0]):
            variable[j] += h[j]
            jac[:, jac_col] = (func(*variables) - f0) / h[j]
            variable[j] -= h[j]
            jac_col += 1
    return jac


def diff_3point(jac_shape, func: callable, *variables):
    """
    3-point numeric finite difference. f'(x) = (f(x+h) - f(x-h)) / 2h
    :param jac_shape: tuple
        the shape of returned jacobian matrix.
    :param func: callable
        the function to evaluate jacobian matrix.
    :param variables: ndarrays
        At which the jacobian matrix is evaluated.
    :return:
    """
    jac = np.zeros(jac_shape, dtype=np.float64)

    jac_col = 0
    for variable in variables:
        h = np.maximum(ROUNDING_ERROR * variable, ROUNDING_ERROR)
        for j in range(h.shape[0]):
            variable[j] += h[j]
            f_plus = func(*variables)
            variable[j] -= 2 * h[j]
            f_subs = func(*variables)
            jac[:, jac_col] = (f_plus - f_subs) / (2 * h[j])
            variable[j] += h[j]
            jac_col += 1
    return jac

def diff_auto(func: Callable[[np.ndarray], np.ndarray], variable: np.ndarray):
    """
    auto diff, only one vector input acceptable
    """
    return np.array(jacobian(func, variable)[1], dtype=np.float64)