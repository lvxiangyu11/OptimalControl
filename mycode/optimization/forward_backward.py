import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 定义微分方程的函数：dx/dt = f(x)
def f(x):
    return 0.5 * x * (1 - x / 3)

# 真实解（用于对比）
def true_solution(t):
    x0 = 0.5  # 初始条件
    C = (3 - x0) / x0
    return 3 / (1 + C * np.exp(-0.5 * t))

# 前向欧拉法
def forward_euler(x0, t_end, h):
    t = np.arange(0, t_end, h)
    x = np.zeros_like(t)
    x[0] = x0
    for i in range(1, len(t)):
        x[i] = x[i - 1] + h * f(x[i - 1])
    return t, x

# 后向欧拉法（解析解二次方程）
def backward_euler(x0, t_end, h):
    t = np.arange(0, t_end, h)
    x = np.zeros_like(t)
    x[0] = x0
    for i in range(1, len(t)):
        a = h / 6
        b = 1 - h / 2
        c = -x[i - 1]
        discriminant = b**2 - 4 * a * c
        if discriminant >= 0:
            x[i] = (-b + np.sqrt(discriminant)) / (2 * a)
        else:
            x[i] = x[i - 1]  # 如果无实根，保持前一个值
    return t, x

# 创建动画函数
def create_euler_animation(x0=0.5, t_end=20, h_forward=0.5, h_backward=0.5):
    t_true = np.linspace(0, t_end, 1000)
    x_true = true_solution(t_true)
    t_forward, x_forward = forward_euler(x0, t_end, h_forward)
    t_backward, x_backward = backward_euler(x0, t_end, h_backward)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_true, x_true, 'k-', alpha=0.3, label='真实解')

    forward_line, = ax.plot([], [], 'ro-', label=f'前向欧拉法 (h={h_forward})')
    backward_line, = ax.plot([], [], 'bs-', label=f'后向欧拉法 (h={h_backward})')

    ax.set_title('欧拉方法比较')
    ax.set_xlabel('时间 (t)')
    ax.set_ylabel('x(t)')
    ax.legend()
    ax.grid(True)

    ax.set_xlim(0, t_end)
    y_max = max(np.max(x_forward), np.max(x_backward), np.max(x_true)) * 1.1
    y_min = min(np.min(x_forward), np.min(x_backward), np.min(x_true)) * 0.9
    ax.set_ylim(y_min, y_max)

    def animate(i):
        forward_idx = min(i, len(t_forward))
        backward_idx = min(i, len(t_backward))
        forward_line.set_data(t_forward[:forward_idx], x_forward[:forward_idx])
        backward_line.set_data(t_backward[:backward_idx], x_backward[:backward_idx])
        return forward_line, backward_line

    frames = max(len(t_forward), len(t_backward))
    anim = FuncAnimation(fig, animate, frames=frames, interval=100, blit=True)
    return anim

# 设置步长，选择其中一个动画显示
h_small = 0.2
anim = create_euler_animation(h_forward=h_small, h_backward=h_small)

# 直接显示动画（适用于 VS Code 环境）
plt.show()
