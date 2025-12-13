import numpy as np
import cvxpy as cp
from fs_dynamics_staticps import dynamics  # 原系统（你可能要改）
from cost import QQt, RRt
import control as ctrl

# 假设已有 UR5 正运动学、雅可比矩阵函数
from ur5_kinematics import fknie_6dof, Jacob_cross_SDH


def get_linearized_dynamics(x_ref, u_ref):
    # 这里你需要修改 dynamics 函数或替换为末端误差模型线性化
    # 原 dynamics 是针对原系统状态、输入；你可能要改成末端误差模型 A, B
    _, A, B = dynamics(x_ref, u_ref)
    return A.T, B.T


def solver_mpc_end_effector(e_t, edot_t, e_ref_traj, edot_ref_traj, u_ref_traj, umax, umin, xmin, xmax, T_pred):
    """
    控制末端执行器误差状态：状态 x = [e; edot]，其中 e = p - p_ref, edot = v_e - v_ref
    控制量 u 是末端速度（或末端加速度）输入近似
    """
    ns = e_t.shape[0] + edot_t.shape[0]
    ni = u_ref_traj.shape[0]
    xx_t = np.vstack((e_t, edot_t))

    xx_mpc = cp.Variable((ns, T_pred))
    uu_mpc = cp.Variable((ni, T_pred - 1))

    cost = 0
    constr = []
    constr += [xx_mpc[:, 0] == xx_t]

    for tt in range(T_pred - 1):
        # 状态参考
        x_ref_tt = np.vstack((e_ref_traj[:, tt], edot_ref_traj[:, tt]))
        u_ref_tt = u_ref_traj[:, tt]

        A, B = get_linearized_dynamics(x_ref_tt, u_ref_tt)

        deltax = xx_mpc[:, tt] - x_ref_tt
        deltau = uu_mpc[:, tt] - u_ref_tt

        cost += cp.quad_form(deltax, QQt)
        cost += cp.quad_form(deltau, RRt)

        x_ref_next = np.vstack((e_ref_traj[:, tt + 1], edot_ref_traj[:, tt + 1]))
        constr += [xx_mpc[:, tt + 1] - x_ref_next == A @ deltax + B @ deltau]

        constr += [uu_mpc[:, tt] <= umax,
                   uu_mpc[:, tt] >= umin,
                   xx_mpc[:, tt] <= xmax,
                   xx_mpc[:, tt] >= xmin]

    # 末端时刻
    x_ref_end = np.vstack((e_ref_traj[:, T_pred - 1], edot_ref_traj[:, T_pred - 1]))
    A_end, B_end = get_linearized_dynamics(x_ref_end, u_ref_traj[:, T_pred - 1])
    QQT = ctrl.dare(A_end, B_end, QQt, RRt)[0]

    cost += cp.quad_form(xx_mpc[:, T_pred - 1] - x_ref_end, QQT)
    constr += [xx_mpc[:, T_pred - 1] <= xmax,
               xx_mpc[:, T_pred - 1] >= xmin]

    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)

    return uu_mpc[:, 0].value, xx_mpc.value, uu_mpc.value


# 主仿真循环
def run_ur5_end_effector_tracking(theta0, traj_func, T_total, dt, T_pred, umax, umin, xmin, xmax):
    theta = theta0.copy()  # 6×1 初始关节角
    N = int(T_total / dt)

    # 记录
    E = np.zeros((N, 3))
    time = np.arange(N) * dt

    p_prev = None
    v_est = np.zeros(3)

    for k in range(N):
        t = k * dt
        # 获取参考末端位置 & 速度（三维向量）
        p_ref, v_ref = traj_func(t)

        # 当前末端位置
        p = fknie_6dof(theta)  # 返回 3×1
        if p_prev is None:
            v_est = np.zeros(3)
        else:
            v_est = (p - p_prev) / dt

        e = p - p_ref
        edot = v_est - v_ref

        # 生成未来 T_pred 参考序列（你需要实现）
        # e_ref_traj: 3×T_pred, edot_ref_traj:3×T_pred, u_ref_traj:3×(T_pred-1)
        e_ref_traj, edot_ref_traj, u_ref_traj = generate_future_refs(t, traj_func, T_pred, dt)

        # 调用 MPC
        u_next, xx_pred, uu_pred = solver_mpc_end_effector(e, edot, e_ref_traj, edot_ref_traj, u_ref_traj, umax, umin,
                                                           xmin, xmax, T_pred)

        # 转换末端速度输入 u_next（3×1）→ 关节角速度 theta_dot（6×1）
        J = Jacob_cross_SDH(theta)  # 6×6
        Jv = J[0:3, :]  # 3×6
        # 伪逆
        theta_dot = np.linalg.pinv(Jv) @ u_next

        # 更新关节角度
        theta = theta + theta_dot * dt

        # 记录误差
        E[k, :] = e.flatten()
        p_prev = p.copy()

    # 绘图，你保持你原可视化格式
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(311)
    plt.plot(time, E[:, 0]);
    plt.ylabel('e_x');
    plt.title('末端误差 x 轴')
    plt.subplot(312)
    plt.plot(time, E[:, 1]);
    plt.ylabel('e_y');
    plt.title('末端误差 y 轴')
    plt.subplot(313)
    plt.plot(time, E[:, 2]);
    plt.ylabel('e_z');
    plt.title('末端误差 z 轴')
    plt.xlabel('Time [s]')
    plt.show()
