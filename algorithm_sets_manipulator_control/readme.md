# PID控制

PID控制的核心是基于误差的三个部分：比例（Proportional）、积分（Integral）和微分（Derivative），分别处理当前误差、累积误差和误差变化率。PID控制的优点在于简单、鲁棒性强且易于调参，但缺点是对于非线性或时变系统可能需要高级扩展（如自适应PID）。

PID控制基于误差 $e(t)=r(t)−y(t)$（设定值减实际输出）生成控制信号 $u(t)$，使系统输出趋近设定值。

$$
u(t)=K_p e(t)+K_i \int_0^t e(\tau) d\tau +K_d \frac{de(t)}{dt}
$$

# 指数趋近律的滑模控制

滑模控制（Sliding Mode Control, SMC）是一种非线性鲁棒控制方法，适用于不确定性系统（如参数变化、外部扰动）。其核心是设计滑动面（sliding surface）$s(x)$ ，使系统状态从初始位置趋近并保持在 $s=0$ 上，从而实现期望动态。指数趋近律（Exponential Reaching Law）是设计切换控制律的一种常见方法，能减小抖振（chattering）并加速趋近过程。

## 理论基础

考虑二阶系统：$\ddot{x} = f(x,\dot{x})+bu+d$ 其中 $f$ 为已知动态， $b$ 为增益, $d$ 为扰动。

定义滑动变量：

$$
s = \dot{e} + ce
$$

其中 $e = x - x_d$ 。

**控制目标**：使 $s \rightarrow 0$ 并保持 $\dot{s} = 0$ 。

## 趋近律设计

**传统等速趋近律**：

$$
\dot{s} = -\epsilon \cdot \text{sgn}(s) \quad (\epsilon > 0)
$$

**指数趋近律**：

$$
\dot{s} = -\epsilon \cdot \text{sgn}(s) - ks, \quad \epsilon > 0, k > 0
$$

- **切换项** $-\epsilon \cdot \text{sgn}(s)$ ：确保趋近（克服扰动）
- **比例项** $-ks$ ：使趋近呈指数形式（快速初始响应）

## 控制率推导

从系统动态：

$$
\dot{s} = \ddot{e} + c\dot{e} = (\ddot{x} - \ddot{x}_d) + c\dot{e} = f + bu + d - \ddot{x}_d + c\dot{e}
$$

设 $\dot{s} = -\epsilon \cdot \text{sgn}(s) - ks$，则控制输入为：

$$
u = b^{-1} \left( -f + \ddot{x}_d - c\dot{e} - \epsilon \cdot \text{sgn}(s) - ks - \hat{d} \right)
$$

其中 $\hat{d}$ 为扰动估计值。

## 优缺点分析

**优点**：
- 快速趋近滑动面
- 减小抖振
- 对不确定性和扰动鲁棒性强
- 适用于机器人和电机等非线性系统

**缺点**：
- 可能仍有轻微抖振需额外缓解
- 参数调优复杂易导致不稳定
- 可能放大噪声

# 模型预测控制（MPC）

MPC是一种基于优化和预测的先进控制算法，适用于多变量、约束系统，如化工过程、机器人路径规划。它使用系统模型预测未来行为，在线求解优化问题生成控制输入，使输出跟踪设定值，同时处理约束。相较PID，MPC更智能但计算密集。

## 核心原理

在每个采样时刻 $k$：
1. 预测未来 $N_p$ 步（预测视界）的系统行为
2. 优化未来 $N_u$ 步（控制视界）的控制序列
3. 只施加第一个控制量 $u(k)$
4. 滚动优化（Receding Horizon）

## 系统模型

假设离散状态空间模型：

$$
\begin{aligned}
x(k+1) &= Ax(k) + Bu(k) \\
y(k) &= Cx(k)
\end{aligned}
$$

基于当前状态 $x(k)$ 的预测：

$$
\begin{aligned}
x(k+1|k) &= Ax(k) + Bu(k|k) \\
x(k+2|k) &= A^2x(k) + ABu(k|k) + Bu(k+1|k) \\
&\vdots \\
x(k+N_p|k) &= A^{N_p}x(k) + \sum_{j=0}^{N_p-1} A^{N_p-1-j}Bu(k+j|k)
\end{aligned}
$$

## 优化目标

最小化目标函数：

$$
J(k) = \sum_{i=1}^{N_p} \| y(k+i|k) - r(k+i) \|^2_Q + \sum_{i=0}^{N_u-1} \| \Delta u(k+i|k) \|^2_R
$$

其中：
- $Q, R$ 为权重矩阵
- $\Delta u(k+i) = u(k+i) - u(k+i-1)$（控制增量，用于平滑控制）

## 约束处理

典型约束：

$$
\begin{aligned}
u_{\min} &\leq u(k+i) \leq u_{\max} \\
|\Delta u| &\leq \Delta u_{\max}
\end{aligned}
$$

## 参数选择

- **$N_p$（预测步长）**：较大值预测更远但计算量更大
- **$N_u$（控制步长）**：较小值简化优化问题

# 代码介绍

## 滑模控制相关代码（Matlab）

1. **`Jacob_cross_SDH.m`**：求解速度雅可比矩阵，采用SDH方法建模
2. **`Trajectory.m`**：基于七段S曲线进行轨迹规划，实现等速趋近律的滑模控制器
3. **`fknie_4dof.m`**：正运动学求解函数
4. **`ikine_nm.m`**：基于牛顿迭代法的逆运动学求解函数
5. **`NewtonMethod_FDiff.m`**：逆运动学求解的辅助函数，用于建立牛顿迭代方程

## 模型预测控制相关代码()

1. **`main.py`**：-运行轨迹生成和跟踪的主脚本。
2. **`animation.py`**：显示最优轨迹和参考轨迹
3. **`armijo_rule.py`**：LQR控制器实现
4. **`cost.py`**：MPC控制器实现（基于预测模型和优化）
5. **`find_equilibria.py`**：解决平衡点数值使用符号系统定义
6. **`fs_dynamics_staticps.py`**：-核心模块定义系统动力学和计算其梯度
7. **`mpc_controller.py`**：使用CVXPY制定并解决一个受限的MPC问题。
8. **`ref_traj_gen.py`**：使用五次多项式生成光滑的参考轨迹。
9. **`solver_LQR.py`**：通过反向Riccati递归计算时变LQR增益。
# 环境要求

- 必要依赖包：
  - Python 3.8+
  - NumPy
  - SciPy
  - SymPy
  - Matplotlib
  - CVXPY
  - Control

# 项目要求

1. **学习并掌握控制算法原理**
2. **复现相关代码实验并了解参数意义**

## 学习建议

### 第一阶段：PID控制基础
- 理解比例、积分、微分项的物理意义
- 掌握参数调优方法（试凑法、Ziegler-Nichols法等）
- 分析参数对系统性能（超调量、调节时间、稳态误差）的影响

### 第二阶段：滑模控制进阶
- 理解滑动模态的概念和设计方法
- 掌握指数趋近律的设计原理
- 分析抖振产生机理及抑制方法

### 第三阶段：MPC高级应用
- 理解预测模型的作用
- 掌握优化问题的构建方法
- 学习约束处理技术
- 比较MPC与LQR的性能差异

通过系统学习这三个控制方法，可以建立起从经典控制到现代控制的完整知识体系。