#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
线性二次调节器(LQR)控制系统的Python实现
"""

import numpy as np
import matplotlib.pyplot as plt
import control  # 用于离散动态系统控制，替代Julia的ControlSystems包
import matplotlib

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决保存图像负号'-'显示为方块的问题

# 时间步长
h = 0.1  
# 系统矩阵定义
A = np.array([[1, h], [0, 1]])  # 状态转移矩阵
B = np.array([[0.5*h*h], [h]])  # 控制输入矩阵

n = 2      # 状态变量数量
m = 1      # 控制输入数量
Tfinal = 10.0  # 最终时间
N = int(Tfinal/h) + 1  # 时间步数
thist = np.arange(0, h*(N-1) + h/2, h)  # 时间序列

# 初始条件
x0 = np.array([[1.0], [0]])

# 成本权重矩阵
Q = np.eye(2)  # 状态成本权重矩阵
R = np.array([[0.1]])  # 控制成本权重矩阵 (注意：这里需要是矩阵而不是标量)
Qn = np.eye(2)  # 最终状态成本权重矩阵

# 成本函数定义
def J(xhist, uhist):
    """
    计算控制过程的总成本
    
    参数:
    xhist -- 状态历史记录，维度为(n,N)
    uhist -- 控制输入历史记录，维度为(m,N-1)
    
    返回:
    cost -- 总成本值
    """
    # 确保最终状态是列向量形式
    final_state = xhist[:, -1].reshape(-1, 1)
    cost = 0.5 * final_state.T @ Qn @ final_state
    
    for k in range(N-1):
        # 确保状态和控制输入是列向量形式
        x_k = xhist[:, k].reshape(-1, 1)
        u_k = uhist[:, k].reshape(-1, 1)
        
        # 计算状态成本和控制成本
        state_cost = 0.5 * x_k.T @ Q @ x_k
        control_cost = 0.5 * u_k.T @ R @ u_k
        
        # 累加成本
        cost = cost + state_cost + control_cost
    
    # 返回标量值
    return cost.item()

# 初始化Riccati方程解矩阵P和控制增益矩阵K
P = np.zeros((n, n, N))
K = np.zeros((m, n, N-1))

# 设置终端成本矩阵
P[:, :, N-1] = Qn

# 后向Riccati递归求解
for k in range(N-2, -1, -1):
    # 计算增益矩阵K
    temp = R + B.T @ P[:, :, k+1] @ B
    K[:, :, k] = np.linalg.inv(temp) @ (B.T @ P[:, :, k+1] @ A)
    
    # 更新Riccati方程解P
    P[:, :, k] = Q + A.T @ P[:, :, k+1] @ (A - B @ K[:, :, k])

# 前向迭代，从初始状态x0开始
xhist = np.zeros((n, N))
xhist[:, 0:1] = x0
uhist = np.zeros((m, N-1))

for k in range(N-1):
    # 计算控制输入
    uhist[:, k:k+1] = -K[:, :, k] @ xhist[:, k:k+1]
    
    # 更新状态
    xhist[:, k+1:k+2] = A @ xhist[:, k:k+1] + B @ uhist[:, k:k+1]

# 计算总成本
cost = J(xhist, uhist)
print(f"总成本: {cost}")

# 创建图形窗口
plt.figure(figsize=(12, 9))

# 绘制状态变量随时间的变化
plt.subplot(2, 2, 1)
plt.plot(thist, xhist[0, :], 'b-', label='位置')
plt.plot(thist, xhist[1, :], 'r--', label='速度')
plt.xlabel('时间 (s)')
plt.ylabel('状态值')
plt.title('状态变量随时间变化')
plt.legend()
plt.grid(True)

# 绘制控制输入随时间的变化
plt.subplot(2, 2, 2)
plt.plot(thist[:-1], uhist[0, :], 'g-', label='控制输入')
plt.xlabel('时间 (s)')
plt.ylabel('控制值')
plt.title('控制输入随时间变化')
plt.legend()
plt.grid(True)

# 带有随机噪声的前向迭代
xhist_noise = np.zeros((n, N))
xhist_noise[:, 0:1] = x0  # 可以设置为任意初始状态
uhist_noise = np.zeros((m, N-1))

for k in range(N-1):
    uhist_noise[:, k:k+1] = -K[:, :, k] @ xhist_noise[:, k:k+1]
    
    # 添加系统噪声 (确保噪声是列向量)
    noise = 0.01 * np.random.randn(2, 1)
    xhist_noise[:, k+1:k+2] = A @ xhist_noise[:, k:k+1] + B @ uhist_noise[:, k:k+1] + noise

# 绘制带噪声的状态变量
plt.subplot(2, 2, 3)
plt.plot(thist, xhist_noise[0, :], 'b-', label='带噪声位置')
plt.plot(thist, xhist_noise[1, :], 'r--', label='带噪声速度')
plt.xlabel('时间 (s)')
plt.ylabel('状态值')
plt.title('带噪声的状态变量')
plt.legend()
plt.grid(True)

# 显示控制增益随时间变化
plt.subplot(2, 2, 4)
plt.plot(range(N-1), K[0, 0, :], 'b-', label='K1')
plt.plot(range(N-1), K[0, 1, :], 'r--', label='K2')
plt.xlabel('时间步')
plt.ylabel('增益值')
plt.title('控制增益随时间变化')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('lqr_control_results.png', dpi=300)
plt.show()

# 计算无穷时域LQR解
# 使用control包替代Julia的ControlSystems.jl
Kinf = control.dlqr(A, B, Q, R)[0]
print(f"无穷时域增益矩阵 Kinf:\n{Kinf}")

# 比较第一个时间步的K和无穷时域K
diff = K[:, :, 0] - Kinf
print(f"K[:,:,0] - Kinf = \n{diff}")

# 使用恒定K的前向迭代
xhist_inf = np.zeros((n, N))
xhist_inf[:, 0:1] = x0
uhist_inf = np.zeros((m, N-1))

for k in range(N-1):
    uhist_inf[:, k:k+1] = -Kinf @ xhist_inf[:, k:k+1]
    xhist_inf[:, k+1:k+2] = A @ xhist_inf[:, k:k+1] + B @ uhist_inf[:, k:k+1]

# 闭环特征值
eig_vals = np.linalg.eigvals(A - B @ Kinf)
print(f"闭环特征值: {eig_vals}")

# 创建新图形窗口，比较有限时域和无穷时域的控制结果
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(thist, xhist[0, :], 'b-', label='有限时域 - 位置')
plt.plot(thist, xhist_inf[0, :], 'g--', label='无穷时域 - 位置')
plt.xlabel('时间 (s)')
plt.ylabel('位置')
plt.title('有限时域与无穷时域位置对比')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(thist[:-1], uhist[0, :], 'r-', label='有限时域 - 控制')
plt.plot(thist[:-1], uhist_inf[0, :], 'm--', label='无穷时域 - 控制')
plt.xlabel('时间 (s)')
plt.ylabel('控制输入')
plt.title('有限时域与无穷时域控制对比')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('lqr_finite_vs_infinite.png', dpi=300)
plt.show()