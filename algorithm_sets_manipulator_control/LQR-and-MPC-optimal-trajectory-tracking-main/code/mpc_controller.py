import numpy as np
import cvxpy as cp
from fs_dynamics_staticps import dynamics 
from cost import QQt, RRt
import control as ctrl

def get_linearized_dynamics(x_ref, u_ref):
    _, A, B = dynamics(x_ref, u_ref)  # Extract A and B from system dynamics
    return A.T, B.T

def solver_mpc(xx_t, x_ref_traj, u_ref_traj, umax, umin, xmin, xmax, T_pred):
    ns = xx_t.shape[0]  # Number of state variables
    ni = u_ref_traj.shape[0]  # Number of control inputs
    
    # Define optimization variables
    xx_mpc = cp.Variable((ns, T_pred))  # Predicted state trajectory
    uu_mpc = cp.Variable((ni, T_pred-1))  # Predicted control trajectory
    
    # Define cost and constraints
    cost = 0
    constr = []
    constr += [xx_mpc[:, 0] == xx_t]  # Initial state constraint
   
    for tt in range(T_pred - 1):
        # Retrieve precomputed A and B
        A, B = get_linearized_dynamics(x_ref_traj[:,tt], u_ref_traj[:,tt])
        # Add costs for state tracking and control effort
        deltax = xx_mpc[:, tt] - x_ref_traj[:, tt]
        deltau = uu_mpc[:, tt] - u_ref_traj[:, tt]
        cost += cp.quad_form(deltax, QQt)
        cost += cp.quad_form(deltau, RRt)
        
        # System dynamics constraint
        deltax_1 = (xx_mpc[:, tt+1]-x_ref_traj[:, tt+1])

        constr += [deltax_1 == A @ deltax + B @ deltau] #deltax=xx_mpc-xx_ref and uu_mpc-uu_ref
        
        # Input and state constraints
        constr += [uu_mpc[:, tt] <= umax,
                   uu_mpc[:, tt] >= umin,
                   xx_mpc[:, tt] <= xmax,
                   xx_mpc[:, tt] >= xmin]
        
    A, B = get_linearized_dynamics(x_ref_traj[:,T_pred-1], u_ref_traj[:,T_pred-1])
    QQT = ctrl.dare(A,B, QQt, RRt)[0]
    # Terminal cost and constraints
    cost += cp.quad_form(xx_mpc[:, T_pred-1] - x_ref_traj[:, T_pred-1], QQT)
    constr += [xx_mpc[:, T_pred-1] <= xmax,
               xx_mpc[:, T_pred-1] >= xmin]
    
    # Solve optimization problem
    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    
    return uu_mpc[:, 0].value, xx_mpc.value, uu_mpc.value
