import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

# UR5 DH parameters
a = [0, -0.425, -0.39225, 0, 0, 0]
d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]

# Forward kinematics to get x y z
def fk(q):
    T = np.eye(4)
    for i in range(6):
        ct = np.cos(q[i])
        st = np.sin(q[i])
        ca = np.cos(alpha[i])
        sa = np.sin(alpha[i])
        T = T @ np.array([[ct, -st*ca, st*sa, a[i]*ct],
                          [st, ct*ca, -ct*sa, a[i]*st],
                          [0, sa, ca, d[i]],
                          [0, 0, 0, 1]])
    return T[:3, 3]

# Simple IK using optimization (for demonstration; in practice, use analytical IK for UR5)
def ik(position):
    def cost(q):
        return np.sum((fk(q) - position)**2)
    res = minimize(cost, np.zeros(6), bounds=[(-2*np.pi, 2*np.pi)]*6)
    return res.x

# Simple dynamics model (assuming independent joints with unit inertia for simplification)
I = np.ones(6)

def dynamics(x, t_idx, q_des, dq_des, ddq_des, use_pid=False):
    # Use t_idx as index since t is discrete now
    q = x[0:6]
    dq = x[6:12]
    if use_pid:
        integral_e = x[12:18]
    q_d = q_des[t_idx]
    dq_d = dq_des[t_idx]
    ddq_d = ddq_des[t_idx]
    # Feedforward
    tau_ff = I * ddq_d
    # Error
    e = q - q_d
    de = dq - dq_d
    if use_pid:
        # PID gains (tuned manually; in practice, optimize)
        Kp = 10.0 * np.ones(6)
        Ki = 1.0 * np.ones(6)
        Kd = 5.0 * np.ones(6)
        u = -Kp * e - Ki * integral_e - Kd * de
        d_integral = e  # Derivative of integral_e
    else:
        # Simple MPC approximation per joint (horizon=1, for demonstration)
        r = 0.01  # Control cost weight
        u = np.zeros(6)
        dt = 0.01  # Prediction step
        for j in range(6):
            predicted_e = e[j] + dt * de[j] + (dt**2 / 2) * ddq_d[j]
            u[j] = - predicted_e / (dt**2 / 2 + r)
        d_integral = np.zeros(6)  # Not used for MPC
    tau = tau_ff + u * I
    ddq = tau / I
    dx = np.concatenate((dq, ddq, d_integral if use_pid else []))
    return dx

# Define desired trajectory functions (line in Cartesian space)
def pos_des_func(t):
    x = 0.5 + 0.01 * t
    y = 0.0 + 0.01 * t
    z = 0.5 + 0.01 * t
    return np.array([x, y, z])

# Simulation time
t = np.linspace(0, 10, 1000)
dt = t[1] - t[0]

# Precompute desired trajectories to speed up
pos_des = np.array([pos_des_func(ti) for ti in t])
q_des = np.array([ik(p) for p in pos_des])
dq_des = np.zeros_like(q_des)
ddq_des = np.zeros_like(q_des)

# Finite differences for derivatives
for i in range(1, len(t)):
    dq_des[i] = (q_des[i] - q_des[i-1]) / dt
for i in range(1, len(t)):
    ddq_des[i] = (dq_des[i] - dq_des[i-1]) / dt

# Initial state (close to starting q_des)
q0 = q_des[0] + 0.01 * np.random.randn(6)
dq0 = np.zeros(6)
integral0 = np.zeros(6)
x0_pid = np.concatenate((q0, dq0, integral0))
x0_mpc = np.concatenate((q0, dq0))  # MPC doesn't use integral

# Simulate for PID (extended state)
def pid_wrapper(x, t_idx):
    return dynamics(x, int(t_idx / dt), q_des, dq_des, ddq_des, use_pid=True)
x_pid = odeint(pid_wrapper, x0_pid, t)

# Simulate for MPC (original state)
def mpc_wrapper(x, t_idx):
    return dynamics(x, int(t_idx / dt), q_des, dq_des, ddq_des, use_pid=False)
x_mpc = odeint(mpc_wrapper, x0_mpc, t)

# Compute positions
pos_pid = np.array([fk(x) for x in x_pid[:, 0:6]])
pos_mpc = np.array([fk(x) for x in x_mpc[:, 0:6]])

# Errors
error_pid = pos_des - pos_pid
error_mpc = pos_des - pos_mpc

# Plot errors on x, y, z axes
fig, axs = plt.subplots(3, 2, figsize=(10, 10))
axs[0, 0].plot(t, error_pid[:, 0], label='x error')
axs[0, 0].set_title('PID x error')
axs[1, 0].plot(t, error_pid[:, 1], label='y error')
axs[1, 0].set_title('PID y error')
axs[2, 0].plot(t, error_pid[:, 2], label='z error')
axs[2, 0].set_title('PID z error')

axs[0, 1].plot(t, error_mpc[:, 0], label='x error')
axs[0, 1].set_title('MPC x error')
axs[1, 1].plot(t, error_mpc[:, 1], label='y error')
axs[1, 1].set_title('MPC y error')
axs[2, 1].plot(t, error_mpc[:, 2], label='z error')
axs[2, 1].set_title('MPC z error')

plt.tight_layout()
plt.show()