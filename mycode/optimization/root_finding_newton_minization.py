import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import solve
import matplotlib.cm as cm
import matplotlib as mpl

# 设置字体参数，解决负号问题（不再设置中文字体）
mpl.rcParams['axes.unicode_minus'] = False

# 定义函数和其导数
def f(x):
    return x**4 + x**3 - x**2 - x

def grad_f(x):
    return 4.0*x**3 + 3.0*x**2 - 2.0*x - 1.0

def hessian_f(x):
    return 12.0*x**2 + 6.0*x - 2.0

# 创建绘图数据
x = np.linspace(-1.75, 1.25, 1000)
y = f(x)

# 设置绘图风格
plt.style.use('seaborn-v0_8')
plt.figure(figsize=(12, 8))

# 基本牛顿法
def newton_step(x0):
    return x0 - grad_f(x0) / hessian_f(x0)

# 正则化牛顿法
def regularized_newton_step(x0):
    beta = 1.0
    H = hessian_f(x0)
    while H <= 0:
        H += beta
        beta *= 2
    return x0 - grad_f(x0) / H

# 函数图像
plt.subplot(2, 2, 1)
plt.plot(x, y, 'b-', linewidth=2, label='f(x) = x⁴ + x³ - x² - x')
plt.grid(True, alpha=0.3)
plt.title('Function f(x) = x⁴ + x³ - x² - x', fontsize=14)
plt.legend()

# 基本牛顿法迭代
plt.subplot(2, 2, 2)
plt.plot(x, y, 'b-', linewidth=2)
xguess = [0.0]
colors = cm.rainbow(np.linspace(0, 1, 6))
for i in range(5):
    plt.plot(xguess[-1], f(xguess[-1]), 'o', markersize=8, color=colors[i], 
             label=f'Iteration {i+1}: x = {xguess[-1]:.4f}')
    xnew = newton_step(xguess[-1])
    xguess.append(xnew)
plt.grid(True, alpha=0.3)
plt.title('Basic Newton\'s Method', fontsize=14)
plt.legend()

# 正则化牛顿法迭代
plt.subplot(2, 2, 3)
plt.plot(x, y, 'b-', linewidth=2)
xguess_reg = [0.0]
for i in range(5):
    plt.plot(xguess_reg[-1], f(xguess_reg[-1]), 's', markersize=8, color=colors[i], 
             label=f'Iteration {i+1}: x = {xguess_reg[-1]:.4f}')
    xnew = regularized_newton_step(xguess_reg[-1])
    xguess_reg.append(xnew)
plt.grid(True, alpha=0.3)
plt.title('Regularized Newton\'s Method', fontsize=14)
plt.legend()

# 方法比较图
plt.subplot(2, 2, 4)
plt.plot(range(len(xguess)), [f(x) for x in xguess], 'ro-', label='Basic Newton')
plt.plot(range(len(xguess_reg)), [f(x) for x in xguess_reg], 'bs-', label='Regularized Newton')
plt.grid(True, alpha=0.3)
plt.title('Convergence of Function Value', fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Function Value')
plt.legend()

plt.tight_layout()
plt.suptitle('Visualization of Newton Optimization Methods', fontsize=16, y=1.02)
plt.show()

# 打印最终结果
print(f"基本牛顿法最终结果: x = {xguess[-1]:.6f}, f(x) = {f(xguess[-1]):.6f}")
print(f"正则化牛顿法最终结果: x = {xguess_reg[-1]:.6f}, f(x) = {f(xguess_reg[-1]):.6f}")
