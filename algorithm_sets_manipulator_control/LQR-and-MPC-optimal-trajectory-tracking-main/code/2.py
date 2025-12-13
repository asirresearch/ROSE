import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cvxpy as cp
import control as ctrl

# UR5 DH parameters
a = [0, -0.425, -0.39225, 0, 0, 0]
d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
alpha = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]

# Forward kinematics to get x y z
def fk(q):
    T = np.eye(4)
    for i in range(6):
        ct = np.cos(q[i])
        st = np.sin(q[i])
        ca = np.cos(alpha[i])
        sa = np.sin(alpha[i])
        T = T @ np.array([[ct, -st * ca, st * sa, a[i] * ct],
                          [st, ct * ca, -ct * sa, a[i] * st],
                          [0, sa, ca, d[i]],
                          [0, 0, 0, 1]])
    return T[:3, 3]

# Simple IK using optimization (for demonstration; in practice, use analytical IK for UR5)
def ik(position):
    def cost(q):
        return np.sum((fk(q) - position) ** 2)
    res = minimize(cost, np.zeros(6), bounds=[(-2 * np.pi, 2 * np.pi)] * 6)
    return res.x

# Simplified dynamics (independent joints, unit inertia, no Coriolis or gravity)
def f(x, u):
    dq = x[6:12]
    ddq = u  # tau = I * ddq, I=1
    return np.concatenate((dq, ddq))

# Continuous-time matrices
n = 6
Acont = np.zeros((2 * n, 2 * n))
Acont[0:n, n:2 * n] = np.eye(n)
Bcont = np.zeros((2 * n, n))
Bcont[n:2 * n, :] = np.eye(n)

# Discrete-time step
dt = 0.01

# Discretized matrices (since MPC assumes discrete-time)
def get_linearized_dynamics(x_ref, u_ref):
    # Linear and independent of ref in this simplified model
    A = np.eye(2 * n) + dt * Acont
    B = dt * Bcont
    return A, B  # Note: Original code had A.T, B.T, but assuming standard here; adjust if needed

# Cost matrices
QQt = np.diag([10.0] * 6 + [10] * 6)  # 更重q误差
RRt = 0.1 * np.eye(6)  # 减小控制惩罚

# MPC solver from the repository (adapted slightly for consistency)
def solver_mpc(xx_t, x_ref_traj, u_ref_traj, umax, umin, xmin, xmax, T_pred):
    ns = xx_t.shape[0]  # 12
    ni = u_ref_traj.shape[0]  # 6

    # Define optimization variables
    xx_mpc = cp.Variable((ns, T_pred))  # Predicted state trajectory
    uu_mpc = cp.Variable((ni, T_pred - 1))  # Predicted control trajectory

    # Define cost and constraints
    cost = 0
    constr = []
    constr += [xx_mpc[:, 0] == xx_t]  # Initial state constraint

    for tt in range(T_pred - 1):
        # Retrieve precomputed A and B
        A, B = get_linearized_dynamics(x_ref_traj[:, tt], u_ref_traj[:, tt])
        # Add costs for state tracking and control effort
        deltax = xx_mpc[:, tt] - x_ref_traj[:, tt]
        deltau = uu_mpc[:, tt] - u_ref_traj[:, tt]
        cost += cp.quad_form(deltax, QQt)
        cost += cp.quad_form(deltau, RRt)

        # System dynamics constraint
        deltax_1 = (xx_mpc[:, tt + 1] - x_ref_traj[:, tt + 1])
        constr += [deltax_1 == A @ deltax + B @ deltau]

        # Input and state constraints
        constr += [uu_mpc[:, tt] <= umax,
                   uu_mpc[:, tt] >= umin,
                   xx_mpc[:, tt] <= xmax,
                   xx_mpc[:, tt] >= xmin]

    # Terminal cost
    A, B = get_linearized_dynamics(x_ref_traj[:, T_pred - 1], u_ref_traj[:, T_pred - 1])
    QQT = ctrl.dare(A, B, QQt, RRt)[0]
    cost += cp.quad_form(xx_mpc[:, T_pred - 1] - x_ref_traj[:, T_pred - 1], QQT)
    constr += [xx_mpc[:, T_pred - 1] <= xmax,
               xx_mpc[:, T_pred - 1] >= xmin]

    # Solve optimization problem
    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)

    return uu_mpc[:, 0].value, xx_mpc.value, uu_mpc.value

# Define desired trajectory functions (sinusoidal in Cartesian space)
def pos_des_func(t):
    amplitude = 0.1
    freq = 2 * np.pi / 10  # Period of 10 seconds
    x = 0.5 + amplitude * np.sin(freq * t)
    y = 0.0 + amplitude * np.sin(freq * t)
    z = 0.5 + amplitude * np.sin(freq * t)
    return np.array([x, y, z])

# Simulation parameters
T = 10  # Total time
num_steps = int(T / dt)
t = np.linspace(0, T, num_steps)

# Precompute desired joint trajectory (approximate)
pos_des = np.array([pos_des_func(ti) for ti in t])
q_des = np.array([ik(p) for p in pos_des])
dq_des = np.zeros_like(q_des)
ddq_des = np.zeros_like(q_des)

# Finite differences for derivatives
for i in range(1, num_steps):
    dq_des[i] = (q_des[i] - q_des[i - 1]) / dt
for i in range(1, num_steps):
    ddq_des[i] = (dq_des[i] - dq_des[i - 1]) / dt

# Constraints
umax = np.ones(6) * 10
umin = -umax
xmax = np.array([np.pi] * 6 + [10] * 6)
xmin = -xmax

# MPC horizon
T_pred = 40

# Initial state for both
initial_q = q_des[0] + 0.01 * np.random.randn(6)
initial_dq = np.zeros(6)
initial_x = np.concatenate((initial_q, initial_dq))

# --- MPC Simulation ---
x_mpc = np.zeros((num_steps, 12))
x_mpc[0] = initial_x

pos_mpc = np.zeros((num_steps, 3))
pos_mpc[0] = fk(x_mpc[0, 0:6])

for k in range(num_steps - 1):
    # Prepare reference trajectories for horizon
    end_idx = min(k + T_pred, num_steps)
    pad = T_pred - (end_idx - k)

    q_ref_h = q_des[k:end_idx]
    dq_ref_h = dq_des[k:end_idx]
    x_ref_traj = np.vstack((q_ref_h.T, dq_ref_h.T))  # (12, T_pred - pad)
    u_ref_traj = ddq_des[k:end_idx].T  # (6, T_pred - pad)

    if pad > 0:
        x_last = x_ref_traj[:, -1:]
        u_last = u_ref_traj[:, -1:]
        x_ref_traj = np.hstack((x_ref_traj, np.tile(x_last, (1, pad))))
        u_ref_traj = np.hstack((u_ref_traj, np.tile(u_last, (1, pad))))

    # Call MPC solver
    u, _, _ = solver_mpc(x_mpc[k], x_ref_traj, u_ref_traj, umax, umin, xmin, xmax, T_pred)

    # Apply control and update state
    dx = f(x_mpc[k], u)
    x_mpc[k + 1] = x_mpc[k] + dt * dx
    pos_mpc[k + 1] = fk(x_mpc[k + 1, 0:6])

# --- PID Simulation ---
x_pid = np.zeros((num_steps, 12))
x_pid[0] = initial_x

pos_pid = np.zeros((num_steps, 3))
pos_pid[0] = fk(x_pid[0, 0:6])

# PID gains (tuned manually; adjust as needed)
Kp = 10.0 * np.ones(6)
Ki = 1.0 * np.ones(6)
Kd = 5.0 * np.ones(6)

# Integral term
integral = np.zeros(6)

for k in range(num_steps - 1):
    q = x_pid[k, 0:6]
    dq = x_pid[k, 6:12]
    q_d = q_des[k]
    dq_d = dq_des[k]
    ddq_d = ddq_des[k]

    # Errors
    e = q - q_d
    de = dq - dq_d

    # Update integral
    integral += e * dt

    # PID control
    u = -Kp * e - Ki * integral - Kd * de

    # Feedforward (optional, for better tracking)
    tau_ff = ddq_d  # Since I=1

    # Total control
    u_total = tau_ff + u

    # Update state
    dx = f(x_pid[k], u_total)
    x_pid[k + 1] = x_pid[k] + dt * dx
    pos_pid[k + 1] = fk(x_pid[k + 1, 0:6])

# Errors
error_mpc = pos_des - pos_mpc
error_pid = pos_des - pos_pid

# Plot errors on x, y, z axes for both PID and MPC
fig, axs = plt.subplots(3, 2, figsize=(10, 10))
axs[0, 0].plot(t, error_pid[:, 0])
axs[0, 0].set_title('PID x error')
axs[1, 0].plot(t, error_pid[:, 1])
axs[1, 0].set_title('PID y error')
axs[2, 0].plot(t, error_pid[:, 2])
axs[2, 0].set_title('PID z error')

axs[0, 1].plot(t, error_mpc[:, 0])
axs[0, 1].set_title('MPC x error')
axs[1, 1].plot(t, error_mpc[:, 1])
axs[1, 1].set_title('MPC y error')
axs[2, 1].plot(t, error_mpc[:, 2])
axs[2, 1].set_title('MPC z error')

plt.tight_layout()
plt.show()