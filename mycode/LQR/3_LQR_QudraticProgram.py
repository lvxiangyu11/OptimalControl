import numpy as np
from scipy import sparse
from scipy.linalg import block_diag
from scipy.sparse import kron, eye
import matplotlib.pyplot as plt
import time

plt.rcParams['font.sans-serif'] = ['SimHei']   
plt.rcParams['axes.unicode_minus'] = False     

# 离散动态系统参数
h = 0.1                  # 时间步长
A = np.array([[1, h], [0, 1]])  # 系统矩阵
B = np.array([[0.5*h*h], [h]])  # 控制矩阵
n = 2                    # 状态数量
m = 1                    # 控制量数量
Tfinal = 100.0           # 最终时间
N = int(Tfinal/h) + 1    # 时间步数

# 创建时间历史数组
times = np.arange(0, h*(N-1) + 0.001, h)  # 加上小量避免浮点精度问题

# 初始条件
x0 = np.array([[1.0], [0]])  # 注意在Python中使用列向量

# 代价权重矩阵（使用稀疏矩阵）
Q = sparse.eye(2)             # 状态代价矩阵
R = 0.1 * sparse.eye(1)       # 控制代价矩阵
Qn = sparse.eye(2)            # 最终状态代价矩阵

# 代价函数定义
def J(xhist, uhist):
    """计算控制系统的总代价"""
    cost = 0.5 * xhist[:, -1].T @ Qn @ xhist[:, -1]
    for k in range(N-1):
        state_cost = 0.5 * xhist[:, k].T @ Q @ xhist[:, k]
        control_cost = 0.5 * uhist[k].T @ R @ uhist[k]
        cost += state_cost + control_cost
    return cost[0, 0]  # 返回标量值

# 构建优化问题的H矩阵（块对角矩阵）
H_blocks = [R]
for _ in range(N-2):
    H_blocks.append(Q)
    H_blocks.append(R)
H_blocks.append(Qn)
H = sparse.block_diag(H_blocks)

# 构建约束矩阵C
B_block = np.hstack((B, -np.eye(2)))
C = kron(sparse.eye(N-1), B_block)

# 在C矩阵中设置A矩阵的位置
C_dense = C.toarray()  # 转为密集矩阵以便更新元素
for k in range(N-2):
    row_idx = (k*n) + np.arange(n)
    col_idx = (k*(n+m)-n) + np.arange(n)
    C_dense[np.ix_(row_idx, col_idx)] = A
C = sparse.csr_matrix(C_dense)  # 转回稀疏矩阵

# 构建右侧向量d
d_top = -A @ x0
d_bottom = np.zeros((C.shape[0]-n, 1))
d = np.vstack((d_top, d_bottom))

# 构建并求解线性系统
H_dense = H.toarray()
C_dense = C.toarray()
zeros_block = np.zeros((C.shape[0], C.shape[0]))
zeros_H = np.zeros((H.shape[0], 1))

left_matrix = np.block([
    [H_dense, C_dense.T],
    [C_dense, zeros_block]
])
right_vector = np.vstack((zeros_H, d))
y = np.linalg.solve(left_matrix, right_vector)

# 提取状态和控制历史
z = y[:H.shape[0]]
Z = z.reshape((n+m, N-1), order='F')  # 注意使用列优先顺序重塑数组

# 获取状态历史
xhist = Z[m:, :]  # 提取状态部分
xhist = np.hstack((x0, xhist))  # 添加初始状态

# 获取控制历史
uhist = Z[:m, :]  # 提取控制部分

# 在每个时间步绘制图形
plt.figure(figsize=(12, 8))
plt.ion()  # 打开交互模式

for k in range(1, N):
    # 清除当前图像
    plt.clf()
    
    # 绘制位置和速度
    plt.subplot(2, 1, 1)
    plt.plot(times[:k], xhist[0, :k], 'b-', label="位置")
    plt.plot(times[:k], xhist[1, :k], 'r-', label="速度")
    plt.xlabel("时间")
    plt.ylabel("状态")
    plt.title(f"状态历史 (t = {times[k-1]:.1f})")
    plt.legend()
    plt.grid(True)
    
    # 绘制控制输入
    plt.subplot(2, 1, 2)
    if k > 1:
        plt.plot(times[:k-1], uhist[0, :k-1], 'g-', label="控制")
    plt.xlabel("时间")
    plt.ylabel("控制输入")
    plt.title("控制历史")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)  # 暂停以便观察
    
    # 可选：保存每一帧
    # plt.savefig(f'frame_{k:04d}.png')

# 计算总代价
total_cost = J(xhist, uhist)
print(f"总代价: {total_cost}")

# 等待用户关闭窗口
plt.ioff()  # 关闭交互模式
plt.show()