import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# -------------------- 中文字体设置 --------------------
plt.rcParams['font.sans-serif'] = ['SimHei']   # 支持中文
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号

# -------------------- UR5 DH 参数 --------------------
a = [0, -0.425, -0.39225, 0, 0, 0]
d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]

def fk(q):
    """UR5 正运动学"""
    T = np.eye(4)
    for i in range(6):
        ct, st = np.cos(q[i]), np.sin(q[i])
        ca, sa = np.cos(alpha[i]), np.sin(alpha[i])
        A_i = np.array([
            [ct, -st*ca, st*sa, a[i]*ct],
            [st, ct*ca, -ct*sa, a[i]*st],
            [0, sa, ca, d[i]],
            [0, 0, 0, 1]
        ])
        T = T @ A_i
    return T[:3, 3]

def ik(position, q0=None):
    """数值逆运动学"""
    if q0 is None:
        q0 = np.zeros(6)
    def cost(q):
        return np.sum((fk(q) - position)**2)
    res = minimize(cost, q0, method='L-BFGS-B', bounds=[(-2*np.pi, 2*np.pi)]*6)
    return res.x

# -------------------- 正弦轨迹 --------------------
A, B, C = 0.1, 0.1, 0.1
omega = 2*np.pi / 10
phi = np.pi/2
dt = 0.01
T_total = 2
t = np.arange(0, T_total, dt)

def traj(t):
    x = A*np.sin(omega*t)
    y = B*np.sin(omega*t + phi)
    z = 1.0 + C*np.sin(omega*t)
    return np.array([x, y, z])

pos_des = np.array([traj(ti) for ti in t])

# -------------------- 逆运动学参考 --------------------
q_des = np.zeros((len(t), 6))
q_prev = np.zeros(6)
for i in range(len(t)):
    q_prev = ik(pos_des[i], q_prev)
    q_des[i] = q_prev

# -------------------- PID 参数 --------------------
Kp = 10.0 * np.ones(6)
Ki = 1.0 * np.ones(6)
Kd = 5.0 * np.ones(6)
umin, umax = -5*np.ones(6), 5*np.ones(6)

# -------------------- 初始化 --------------------
x_pid = np.zeros((len(t), 12))
pos_pid = np.zeros((len(t), 3))
pos_pid[0] = fk(x_pid[0,:6])
integral = np.zeros(6)

# -------------------- PID 仿真 --------------------
for k in range(len(t)-1):
    q, dq = x_pid[k,:6], x_pid[k,6:]
    q_d = q_des[k]
    e = q - q_d
    de = dq - np.zeros(6)
    integral += e*dt
    u = -Kp*e - Ki*integral - Kd*de
    u = np.clip(u, umin, umax)
    dq_next = dq + dt*u
    q_next = q + dt*dq
    x_pid[k+1] = np.concatenate((q_next, dq_next))
    pos_pid[k+1] = fk(q_next)

# -------------------- 末端误差曲线 --------------------
pos_err = pos_pid - pos_des

plt.figure(figsize=(8,5))
plt.plot(t, pos_err[:,0], 'r', label='x error')
plt.plot(t, pos_err[:,1], 'g', label='y error')
plt.plot(t, pos_err[:,2], 'b', label='z error')
plt.xlabel('time [s]')
plt.ylabel('End position error [mm]')
plt.title('PID end position error curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
