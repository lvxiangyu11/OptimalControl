import numpy as np
import matplotlib.pyplot as plt
# 定义系统参数
A = np.array([[0, 1], [-1, -1]])
B = np.array([[0], [1]])
Q = np.eye(2)
R = np.array([[1]])
# 定义扰动参数
Q_d = np.array([[0.1, 0], [0, 0.1]])
# 定义时间参数
T = 10
dt = 0.1
time_steps = int(T / dt)
# 初始化状态和控制输入
x = np.zeros((2, time_steps))
u = np.zeros((1, time_steps))
# Minimax DDP算法实现
for t in range(time_steps - 1):
    # 计算当前状态
    x_current = x[:, t]
    # 计算最优控制输入
    u[t] = -np.linalg.inv(R + B.T @ Q @ B) @ (B.T @ Q @ x_current)
    # 更新状态
    d = np.random.multivariate_normal(np.zeros(2), Q_d)  # 添加扰动
    x[:, t + 1] = x_current + (A @ x_current + B @ u[t] + d) * dt
# 绘制结果
plt.figure(figsize=(10, 5))
plt.plot(np.arange(0, T, dt), x[0, :], label='State x1')
plt.plot(np.arange(0, T, dt), x[1, :], label='State x2')
plt.plot(np.arange(0, T, dt), u[0, :], label='Control Input u')
plt.xlabel('Time (s)')
plt.ylabel('State/Control')
plt.title('Minimax DDP Simulation Results')
plt.legend()
plt.grid()
plt.show()