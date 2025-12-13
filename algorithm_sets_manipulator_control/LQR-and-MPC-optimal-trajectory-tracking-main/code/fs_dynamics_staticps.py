import numpy as np
import sympy as sp

ns = 8
ni = 2
dt = 1e-3

# Dynamic parameters sets
# Set 3 
mm = np.array([0.1, 0.1])
mm_act = np.array([0.4, 0.4])
dd = 0.3
alpha = 128 * 0.15
cc = 0.1

# Symbolic variables for calculating the derivatives
xx = sp.Matrix(sp.symbols('x0:8')) 
uu = sp.Matrix(sp.symbols('u0:2'))

# Acceleration for the 1st point
zz1 = (1/mm[0]) * ( - alpha * ((xx[0]-xx[1])/(dd * (dd**2 - (xx[0]-xx[1])**2))
                            + (xx[0]-xx[2])/(2*dd * (4*dd**2 - (xx[0]-xx[2])**2))
                            + (xx[0]-xx[3])/(3*dd * (9*dd**2 - (xx[0]-xx[3])**2))
                            + (xx[0])/(dd * (dd**2 - (xx[0])**2))
                            + (xx[0])/(4*dd * (16*dd**2 - (xx[0])**2))) - cc * xx[4])

# Acceleration for the 2nd point
zz2 = (1/mm_act[0]) * (uu[0] - alpha * ((xx[1]-xx[0])/(dd * (dd**2 - (xx[1]-xx[0])**2))
                            + (xx[1]-xx[2])/(dd * (dd**2 - (xx[1]-xx[2])**2))
                            + (xx[1]-xx[3])/(2*dd * (4*dd**2 - (xx[1]-xx[3])**2))
                            + (xx[1])/(2*dd * (4*dd**2 - (xx[1])**2))
                            + (xx[1])/(3*dd * (9*dd**2 - (xx[1])**2))) - cc * xx[5])

# Acceleration for the 3rd point
zz3 = (1/mm[1]) * ( - alpha * ((xx[2]-xx[0])/(2*dd * (4*dd**2 - (xx[2]-xx[0])**2))
                            + (xx[2]-xx[1])/(dd * (dd**2 - (xx[2]-xx[1])**2))
                            + (xx[2]-xx[3])/(dd * (dd**2 - (xx[2]-xx[3])**2))
                            + (xx[2])/(3*dd * (9*dd**2 - (xx[2])**2))
                            + (xx[2])/(2*dd * (4*dd**2 - (xx[2])**2))) - cc * xx[6])

# Acceleration for the 4th point
zz4 = (1/mm_act[1]) * (uu[1] - alpha * ((xx[3]-xx[0])/(3*dd * (9*dd**2 - (xx[3]-xx[0])**2))
                            + (xx[3]-xx[1])/(2*dd * (4*dd**2 - (xx[3]-xx[1])**2))
                            + (xx[3]-xx[2])/(dd * (dd**2 - (xx[3]-xx[2])**2))
                            + (xx[3])/(4*dd * (16*dd**2 - (xx[3])**2))
                            + (xx[3])/(1*dd * (dd**2 - (xx[3])**2))) - cc * xx[7])

# Dynamics discretization
dynamics = sp.Matrix([
    xx[0] + dt*xx[4],  
    xx[1] + dt*xx[5],
    xx[2] + dt*xx[6],
    xx[3] + dt*xx[7],
    xx[4] + dt*zz1,    
    xx[5] + dt*zz2,
    xx[6] + dt*zz3,
    xx[7] + dt*zz4
])

# First calculate gradients
dfx_sym = sp.Matrix([[dynamics[j].diff(xx[i]) 
                      for j in range(ns)]  # equation number
                      for i in range(ns)])  # derivative w.r.t. state

dfu_sym = sp.Matrix([[dynamics[j].diff(uu[i]) 
                      for j in range(ns)]  # equation number
                      for i in range(ni)])  # derivative w.r.t. input


# Convert to Python functions
dynamics_func = sp.lambdify((xx, uu), dynamics)
dfx_func = sp.lambdify((xx, uu), dfx_sym)
dfu_func = sp.lambdify((xx, uu), dfu_sym)

def dynamics(xx, uu):
    # Evaluate dynamics and gradients
    xxp = np.array(dynamics_func(xx, uu)).astype(np.float64).squeeze()
    dfx = np.array(dfx_func(xx, uu)).astype(np.float64)
    dfu = np.array(dfu_func(xx, uu)).astype(np.float64)

    return xxp, dfx, dfu
