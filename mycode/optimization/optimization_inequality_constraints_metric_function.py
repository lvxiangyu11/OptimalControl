import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve, norm
from matplotlib.animation import FuncAnimation

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 目标函数和梯度定义
Q = np.diag([0.5, 1.0])

def f(x):
    return 0.5 * (x - np.array([1, 0])).T @ Q @ (x - np.array([1, 0]))

def grad_f(x):
    return Q @ (x - np.array([1, 0]))

def hess_f(x):
    return Q

# 约束函数及其梯度
def c(x):
    return np.array([x[0]**2 + 2*x[0] - x[1]])

def dc(x):
    return np.array([[2*x[0] + 2, -1]])

# 惩罚项定义
rho = 1.0
def P(x, lam):
    return f(x) + lam @ c(x) + 0.5 * rho * norm(c(x))**2

def grad_P(x, lam):
    g = grad_f(x) + dc(x).T @ (lam + rho * c(x))
    return np.hstack([g, c(x)])

# Gauss-Newton步骤
def gauss_newton_step(x, lam):
    H = hess_f(x)
    C = dc(x)
    grad_L = np.hstack([-grad_f(x) - C.T @ lam, -c(x)])
    KKT = np.block([[H, C.T],
                    [C, np.zeros((1, 1))]])
    delta = solve(KKT, grad_L)
    return delta[:2], delta[2:]

# 初始化数据
x_list = [np.array([-1.0, -1.0])]
lam_list = [np.array([0.0])]
max_iter = 20

# 预计算等高线数据
def compute_contour():
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    return X, Y, Z

X, Y, Z = compute_contour()

# 设置图像
fig, ax = plt.subplots()
ax.set_title("Gauss-Newton 方法优化过程（含约束）")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.contour(X, Y, Z, levels=30)
xc = np.linspace(-3.2, 1.2, 100)
yc = xc**2 + 2 * xc
ax.plot(xc, yc, 'y', label='约束曲线')
path_plot, = ax.plot([], [], 'rx-', label='迭代轨迹')
point_plot, = ax.plot([], [], 'ro')
ax.legend()

# 动画每一帧：执行一次迭代
def update(frame):
    if frame >= len(x_list):
        x = x_list[-1]
        lam = lam_list[-1]
        dx, dlam = gauss_newton_step(x, lam)

        # 线性回退
        alpha = 1.0
        while P(x + alpha * dx, lam + alpha * dlam) > \
              P(x, lam) + 0.01 * alpha * grad_P(x, lam) @ np.hstack([dx, dlam]):
            alpha *= 0.5

        x_new = x + alpha * dx
        lam_new = lam + alpha * dlam
        x_list.append(x_new)
        lam_list.append(lam_new)

    # 更新轨迹
    data = np.array(x_list)
    path_plot.set_data(data[:, 0], data[:, 1])
    point_plot.set_data([data[-1][0]], [data[-1][1]]) 
    return path_plot, point_plot

# 动画创建
anim = FuncAnimation(fig, update, frames=max_iter, interval=800, repeat=False)

plt.show()