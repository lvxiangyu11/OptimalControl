import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']   # 使用黑体
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号

# 参数设置
N = 100            # 离散节点数
T = 1.0            # 总时间
h = T / (N - 1)    # 时间步长
x0, v0 = 0, 0      # 初始位置和速度
xf, vf = 1, 0      # 末端位置和速度

# 给定终点协态变量，递推整个轨迹
def simulate(lam_xN, lam_vN):
    lam_x = np.zeros(N)    # 协态变量 lambda_x
    lam_v = np.zeros(N)    # 协态变量 lambda_v
    lam_x[-1] = lam_xN     # 终点边界条件
    lam_v[-1] = lam_vN
    # 反向递推协态变量
    for k in range(N-2, -1, -1):
        lam_x[k] = lam_x[k+1]
        lam_v[k] = lam_v[k+1] + h * lam_x[k+1]
    x = np.zeros(N)        # 位置
    v = np.zeros(N)        # 速度
    u = np.zeros(N-1)      # 控制输入
    x[0] = x0
    v[0] = v0
    # 正向递推状态和控制
    for k in range(N-1):
        u[k] = -0.5 * lam_v[k+1]
        x[k+1] = x[k] + h * v[k]
        v[k+1] = v[k] + h * u[k]
    return x, v, u

# 射击法目标函数：返回终点状态误差
def boundary_error(lamN):
    lam_xN, lam_vN = lamN  # lamN为终点协态变量
    x, v, u = simulate(lam_xN, lam_vN)
    # 返回终点位置和速度的误差
    return np.array([x[-1] - xf, v[-1] - vf])

# 求解使得终点约束成立的协态变量终值
sol = root(boundary_error, [0.0, 0.0])
lam_xN_opt, lam_vN_opt = sol.x
x, v, u = simulate(lam_xN_opt, lam_vN_opt)

# 画图
t = np.linspace(0, T, N)
plt.figure(figsize=(9,5))
plt.plot(t, x, label='位置 $x_k$')
plt.plot(t, v, label='速度 $v_k$')
plt.plot(t[:-1], u, label='控制 $u_k$')
plt.xlabel('时间')
plt.ylabel('数值')
plt.title('离散最优控制轨迹（射击法）')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 打印最终位置和速度，检查是否满足目标约束
print(f"终点位置 x_N = {x[-1]:.6f}, 终点速度 v_N = {v[-1]:.6f}")
print(f"终点协态变量 lam_xN = {lam_xN_opt:.6f}, lam_vN = {lam_vN_opt:.6f}")
