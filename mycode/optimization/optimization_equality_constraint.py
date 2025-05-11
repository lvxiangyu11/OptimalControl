import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 定义目标函数 f(x1, x2) = (x1-2)^2 + (x2-1)^2
def f(x1, x2):
    return (x1-2)**2 + (x2-1)**2

# 定义约束条件 c(x1, x2) = x1 + x2 - 3 = 0
def c(x1, x2):
    return x1 + x2 - 3

# 创建网格
x1 = np.linspace(-1, 5, 100)
x2 = np.linspace(-1, 5, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

# 创建绘图
fig = plt.figure(figsize=(18, 6))

# 3D 曲面图
ax1 = fig.add_subplot(131, projection='3d')
surf = ax1.plot_surface(X1, X2, Z, cmap=cm.coolwarm, alpha=0.7)
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_zlabel('$f(x_1, x_2)$')
ax1.set_title('目标函数')

# 约束线 x1 + x2 = 3
constraint_x1 = np.linspace(0, 3, 100)
constraint_x2 = 3 - constraint_x1
ax1.plot(constraint_x1, constraint_x2, f(constraint_x1, constraint_x2), 'g-', linewidth=3)

# 最优点
opt_x1, opt_x2 = 1.5, 1.5  # 解析解
ax1.scatter([opt_x1], [opt_x2], [f(opt_x1, opt_x2)], color='r', s=100, marker='o')

# 等高线图
ax2 = fig.add_subplot(132)
contour = ax2.contour(X1, X2, Z, 20, cmap=cm.coolwarm)
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_title('等高线与约束条件')
ax2.plot(constraint_x1, constraint_x2, 'g-', linewidth=2, label='约束: $x_1 + x_2 = 3$')
ax2.scatter([opt_x1], [opt_x2], color='r', s=100, marker='o', label='最优点 (1.5, 1.5)')
ax2.legend()

# 梯度可视化
ax3 = fig.add_subplot(133)
contour = ax3.contour(X1, X2, Z, 20, cmap=cm.coolwarm)
ax3.set_xlabel('$x_1$')
ax3.set_ylabel('$x_2$')
ax3.set_title('梯度与约束')
ax3.plot(constraint_x1, constraint_x2, 'g-', linewidth=2)
ax3.scatter([opt_x1], [opt_x2], color='r', s=100, marker='o')

# 绘制梯度向量
gradient_x1 = 2 * (opt_x1 - 2)  # ∂f/∂x1
gradient_x2 = 2 * (opt_x2 - 1)  # ∂f/∂x2
constraint_grad_x1 = 1  # ∂c/∂x1
constraint_grad_x2 = 1  # ∂c/∂x2

scale = 0.5  # 缩放梯度向量以便可视化
ax3.arrow(opt_x1, opt_x2, scale*gradient_x1, scale*gradient_x2, head_width=0.1, head_length=0.1, fc='blue', ec='blue', label='$\\nabla f$')
ax3.arrow(opt_x1, opt_x2, -scale*constraint_grad_x1, -scale*constraint_grad_x2, head_width=0.1, head_length=0.1, fc='red', ec='red', label='$-\\nabla c$')
ax3.legend()

plt.tight_layout()
plt.show()