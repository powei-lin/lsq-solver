# lsq-solver

[![PyPI - Version](https://img.shields.io/pypi/v/lsq-solver.svg)](https://pypi.org/project/lsq-solver)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lsq-solver.svg)](https://pypi.org/project/lsq-solver)
[![PyPI Downloads](https://static.pepy.tech/badge/lsq-solver)](https://pepy.tech/projects/lsq-solver)

-----
`lsq-solver` is a user-friendly wrapper around `scipy.optimize.least_squares` that mimics the Ceres-Solver API, designed to make nonlinear least-squares optimization in Python cleaner and more intuitive.

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install lsq-solver
```

## Usage
```py
import numpy as np
from functools import partial
from lsq_solver import LeastSquaresProblem

def cost_func0(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    residual = ...
    return residual.flatten() # the shape needs to be (n,)

def cost_func1(b: np.ndarray, c: np.ndarray) -> np.ndarray:
    residual = ...
    return residual.flatten()

def main():
    problem = LeastSquaresProblem()
    a = np.array(...) # shape (m,)
    b = np.array(...)
    c = np.array(...)
    
    # add cost_func0, fix a, solve b, c
    residual_num0 = ...
    cost0 = partial(cost_func0, a)
    problem.add_residual_block(residual_num0, cost0, b, c, jac_func="2-point")

    # add cost_func1
    residual_num1 = ...
    problem.add_residual_block(residual_num1, cost_func1, b, c, jac_func="2-point")

    # solve
    problem.solve(verbose=2)

if __name__ == "__main__":
    main()
```

## License

`lsq-solver` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
