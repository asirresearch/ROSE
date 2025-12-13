import numpy as np
import fs_dynamics_staticps as fsd
from sympy import symbols, Matrix, nsolve

# Define symbolic variables for states (xx) and controls (uu)
xx = symbols('x0:8')  # x0, x1, ..., x7 (8 states)
uu_sym = symbols('u0:2')  # u0, u1 (2 controls)

zz1 = fsd.zz1
zz2 = fsd.zz2
zz3 = fsd.zz3
zz4 = fsd.zz4

# Define numerical equilibrium solver
def find_equilibrium_numerical(xx0_values, uu0):
    # Define the residual equations
    residual_eqs = Matrix([xx[4], xx[5], xx[6], xx[7], zz1, zz2, zz3, zz4])
    # Substitute control inputs
    residual_eqs = residual_eqs.subs({uu_sym[0]: uu0[0], uu_sym[1]: uu0[1]})
    # Use nsolve for numerical solution
    xx0_values_sym = Matrix(xx0_values)
    equilibrium = nsolve(residual_eqs, xx, xx0_values_sym)

    return equilibrium, uu0

def to_numpy_array(value):
    # Converts the input value (SymPy Matrix, list, or other iterable) to a NumPy array.
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.array(value, dtype=float)  # Ensure float data type
    elif hasattr(value, "tolist"):  # For SymPy Matrices
        return np.array(value.tolist(), dtype=float)
    else:
        raise ValueError(f"Cannot convert {type(value)} to a NumPy array")
