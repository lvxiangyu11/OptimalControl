# -*- coding: utf-8 -*-
"""
最优控制问题：使用动态规划（DP）和二次规划（QP）方法
"""

import numpy as np
from scipy import sparse
from scipy.linalg import block_diag
from scipy.sparse import kron, eye
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve
import matplotlib.pyplot as plt
from control import dlqr, dare
import numdifftools as nd  # 用于数值梯度计算
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决保存图像负号'-'显示为方块的问题

# 离散动态系统参数
h = 0.1   # 时间步长
A = np.array([[1, h], [0, 1]])  # 状态转移矩阵
B = np.array([[0.5*h*h], [h]])  # 控制输入矩阵

# 检查可控性
controllability = np.hstack([B, A @ B])
print(f"系统可控性矩阵的秩: {np.linalg.matrix_rank(controllability)}")

n = 2      # 状态变量的维度
m = 1      # 控制输入的维度
Tfinal = 10.0  # 仿真结束时间
N = int(Tfinal/h) + 1  # 时间步数
thist = np.arange(0, h*N, h)[:N]  # 时间序列

# 初始状态
x0 = np.array([[1.0], [0.0]])

# 代价函数权重
Q = np.eye(n)  # 状态代价权重
R = 0.1 * np.eye(m)  # 控制代价权重
Qn = np.eye(n)  # 终端状态代价权重

def J(xhist, uhist):
    """
    计算总代价
    
    参数:
    xhist -- 状态历史，形状为 (n, N)
    uhist -- 控制历史，形状为 (m, N-1)
    
    返回值:
    cost -- 总代价（标量）
    """
    # 确保向量的形状正确
    x_end = xhist[:, -1:] if len(xhist.shape) > 1 else xhist.reshape(n, 1)
    
    # 计算终端代价
    cost = 0.5 * x_end.T @ Qn @ x_end
    
    # 计算轨迹代价
    for k in range(xhist.shape[1] - 1):
        x_k = xhist[:, k:k+1] if len(xhist.shape) > 1 else xhist.reshape(n, 1)
        u_k = uhist[:, k:k+1] if len(uhist.shape) > 1 else uhist.reshape(m, 1)
        cost += 0.5 * x_k.T @ Q @ x_k + 0.5 * u_k.T @ R @ u_k
    
    # 确保返回标量值
    if hasattr(cost, "item"):  # 如果是numpy数组或矩阵
        return cost.item()
    else:  # 如果已经是标量
        return cost

# 二次规划（QP）解决方案
# 构建代价矩阵
blocks = [R]
for _ in range(N-2):
    blocks.extend([Q, R])
blocks.append(Qn)
H = block_diag(*blocks)

# 构建约束矩阵
C = np.zeros((n * (N-1), (n+m) * (N-1)))

# 填充约束矩阵
for i in range(N-1):
    # 对于控制项
    C[i*n:(i+1)*n, i*(n+m):i*(n+m)+m] = B
    
    # 对于状态项
    C[i*n:(i+1)*n, i*(n+m)+m:(i+1)*(n+m)] = -np.eye(n)
    
    # 对于之前的状态转移
    if i > 0:
        C[i*n:(i+1)*n, (i-1)*(n+m)+m:i*(n+m)] = A

# 构建右侧向量
d = np.zeros((n * (N-1), 1))
d[0:n] = -A @ x0

# 解线性系统
M = np.block([
    [H, C.T],
    [C, np.zeros((C.shape[0], C.shape[0]))]
])

rhs = np.vstack([
    np.zeros((H.shape[0], 1)),
    d
])

y = solve(M, rhs)

# 获取拉格朗日乘子
λhist_qp = y[H.shape[0]:].reshape(n, N-1, order='F')

# 获取状态历史和控制历史
z = y[:H.shape[0]]
xhist_qp = np.zeros((n, N))
xhist_qp[:, 0] = x0.flatten()
uhist_qp = np.zeros((m, N-1))

for i in range(N-1):
    uhist_qp[:, i] = z[i*(n+m):(i*(n+m)+m)].flatten()
    xhist_qp[:, i+1] = z[i*(n+m)+m:(i+1)*(n+m)].flatten()

# 动态规划解决方案
P = np.zeros((n, n, N))
K = np.zeros((m, n, N-1))

P[:, :, N-1] = Qn

# 向后Riccati递归
for k in range(N-2, -1, -1):
    K_temp = np.linalg.inv(R + B.T @ P[:, :, k+1] @ B) @ (B.T @ P[:, :, k+1] @ A)
    K[:, :, k] = K_temp
    P[:, :, k] = Q + K_temp.T @ R @ K_temp + (A - B @ K_temp).T @ P[:, :, k+1] @ (A - B @ K_temp)

# 从初始状态开始的前向模拟
xhist = np.zeros((n, N))
xhist[:, 0] = x0.flatten()
uhist = np.zeros((m, N-1))

for k in range(N-1):
    uhist[:, k] = -K[:, :, k] @ xhist[:, k:k+1]
    xhist[:, k+1] = (A @ xhist[:, k:k+1] + B @ uhist[:, k:k+1]).flatten()

# 计算无穷时域的K和P矩阵
K_inf_tuple = dlqr(A, B, Q, R)
K_inf = K_inf_tuple[0]  # 提取K矩阵
Pinf = dare(A, B, Q, R)[0]  # 提取P矩阵

# 无穷时域的值函数
def Vinf(x):
    """
    计算无穷时域的值函数
    """
    x_vec = x.reshape(n, 1) if len(x.shape) == 1 else x
    val = 0.5 * x_vec.T @ Pinf @ x_vec
    return val.item() if hasattr(val, "item") else val

# 有限时域的值函数
def V(k, x):
    """
    计算有限时域k处的值函数
    """
    x_vec = x.reshape(n, 1) if len(x.shape) == 1 else x
    val = 0.5 * x_vec.T @ P[:, :, k] @ x_vec
    return val.item() if hasattr(val, "item") else val

# 从k时刻开始的状态轨迹生成函数
def rollout(k, x):
    """
    从时间步k和状态x开始的状态轨迹生成
    """
    xsub = np.zeros((n, N-k))
    xsub[:, 0] = x.flatten()
    usub = np.zeros((m, N-k-1))
    
    for j in range(N-k-1):
        usub[:, j] = -K[:, :, j+k] @ xsub[:, j:j+1]
        xsub[:, j+1] = (A @ xsub[:, j:j+1] + B @ usub[:, j:j+1]).flatten()
    
    return xsub, usub

# 可视化
plt.figure(figsize=(14, 10))

# 状态轨迹对比图
plt.subplot(2, 2, 1)
plt.plot(thist, xhist[0, :], 'b-', label='DP 位置')
plt.plot(thist, xhist[1, :], 'r-', label='DP 速度')
plt.plot(thist, xhist_qp[0, :], 'b--', label='QP 位置')
plt.plot(thist, xhist_qp[1, :], 'r--', label='QP 速度')
plt.xlabel('时间')
plt.ylabel('状态')
plt.legend()
plt.title('状态轨迹对比')
plt.grid(True)

# 控制输入对比图
plt.subplot(2, 2, 2)
plt.plot(thist[:-1], uhist[0, :], 'g-', label='DP 控制')
plt.plot(thist[:-1], uhist_qp[0, :], 'g--', label='QP 控制')
plt.xlabel('时间')
plt.ylabel('控制输入')
plt.legend()
plt.title('控制输入对比')
plt.grid(True)

# P矩阵元素随时间变化图
plt.subplot(2, 2, 3)
plt.plot(range(N), P[0, 0, :], 'b-', label='P[0,0]')
plt.plot(range(N), P[0, 1, :], 'r-', label='P[0,1]')
plt.plot(range(N), P[1, 1, :], 'g-', label='P[1,1]')
plt.xlabel('时间步')
plt.ylabel('P矩阵元素值')
plt.legend()
plt.title('P矩阵元素随时间变化')
plt.grid(True)

# 打印代价函数值
dp_cost = J(xhist, uhist)
qp_cost = J(xhist_qp, uhist_qp)
print(f"动态规划代价: {dp_cost}")
print(f"二次规划代价: {qp_cost}")
print(f"无穷时域K矩阵与有限时域K矩阵第一步的差异:")
print(K[:, :, 0] - K_inf)
print(f"无穷时域P矩阵与有限时域P矩阵第一步的差异:")
print(P[:, :, 0] - Pinf)

# 从中间时刻开始生成子轨迹
k = 50
if k < N-1:  # 确保k在有效范围内
    x_k = xhist[:, k:k+1]
    xsub, usub = rollout(k, x_k)

    # 检查子轨迹是否最优
    print(f"子轨迹与原轨迹的差异（状态）:")
    print(np.max(np.abs(xsub - xhist[:, k:N])))
    print(f"子轨迹与原轨迹的差异（控制）:")
    print(np.max(np.abs(usub - uhist[:, k:N-1])))

    # 比较拉格朗日乘子与值函数梯度
    print(f"时间步{k}的拉格朗日乘子:")
    if k-1 < λhist_qp.shape[1]:
        print(λhist_qp[:, k-1])
    else:
        print("超出范围")

    # 计算有限时域值函数关于状态的梯度
    def V_gradient_k(x, k):
        x_vec = x.reshape(n, 1) if len(x.shape) == 1 else x
        return P[:, :, k] @ x_vec

    dp_gradient = V_gradient_k(xhist[:, k], k).flatten()
    print(f"时间步{k}的值函数梯度（动态规划）:")
    print(dp_gradient)

    # 计算无穷时域值函数关于状态的梯度
    def Vinf_gradient(x):
        x_vec = x.reshape(n, 1) if len(x.shape) == 1 else x
        return Pinf @ x_vec

    inf_gradient = Vinf_gradient(xhist[:, k]).flatten()
    print(f"无穷时域值函数梯度:")
    print(inf_gradient)

    # 使用有限差分计算代价函数关于状态的梯度
    try:
        eps = 1e-6
        x1p, u1p = rollout(k, xhist[:, k:k+1] + np.array([[eps], [0]]))
        x2p, u2p = rollout(k, xhist[:, k:k+1] + np.array([[0], [eps]]))
        
        # 计算有限差分梯度
        J_base = J(xhist[:, k:N], uhist[:, k:N-1])
        J_x1p = J(x1p, u1p)
        J_x2p = J(x2p, u2p)
        
        lambda_fd = np.array([
            (J_x1p - J_base) / eps,
            (J_x2p - J_base) / eps
        ])
        print(f"通过有限差分计算的梯度:")
        print(lambda_fd)

        # 额外添加梯度比较可视化图
        plt.subplot(2, 2, 4)
        methods = ['拉格朗日乘子', 'DP梯度', '无穷时域梯度', '有限差分']
        
        # 准备数据用于绘图
        gradients = []
        
        # 拉格朗日乘子
        if k-1 < λhist_qp.shape[1]:
            gradients.append(λhist_qp[:, k-1].flatten())
        else:
            gradients.append(np.zeros(n))
            
        gradients.extend([dp_gradient, inf_gradient, lambda_fd])
        
        x_positions = []
        for i in range(4):
            x_positions.extend([i*3, i*3+1])
        
        # 创建柱状图
        bars = plt.bar(x_positions, np.concatenate([g for g in gradients]), width=0.7)
        
        # 为不同方法设置不同颜色
        colors = ['blue', 'green', 'red', 'purple']
        for i in range(4):
            bars[i*2].set_color(colors[i])
            bars[i*2+1].set_color(colors[i])
        
        plt.xticks([1.5 + 3*i for i in range(4)], methods)
        plt.ylabel('梯度值')
        plt.title('不同方法计算的梯度对比')
        plt.grid(True)
    except Exception as e:
        print(f"计算有限差分梯度时出错: {e}")
        plt.subplot(2, 2, 4)
        plt.text(0.5, 0.5, f'数据不可用: {e}', ha='center', va='center')
        plt.title('梯度对比 (数据不可用)')

plt.tight_layout()
plt.show()