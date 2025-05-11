import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

# `设置图像样式`
mpl.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

# `定义目标函数及导数`
def f(x):
    return x**4 + x**3 - x**2 - x

def grad_f(x):
    return 4*x**3 + 3*x**2 - 2*x - 1

def hessian_f(x):
    return 12*x**2 + 6*x - 2

# `阻尼牛顿法实现`
def damped_newton_step(x0, beta=1.0):
    H = hessian_f(x0)
    H_damped = H + beta  # `对 Hessian 加上扰动项`
    return x0 - grad_f(x0) / H_damped

# `执行迭代并记录路径`
def run_damped_newton(x0, beta, steps=5):
    path = [x0]
    for _ in range(steps):
        x_new = damped_newton_step(path[-1], beta=beta)
        path.append(x_new)
    return path

# `参数设置`
x0 = 0.0
beta_values = [1.0, 5.0]
colors = cm.rainbow(np.linspace(0, 1, 6))

# `生成路径`
path_beta_1 = run_damped_newton(x0, beta=1.0)
path_beta_5 = run_damped_newton(x0, beta=5.0)

# `绘图：函数图像与迭代轨迹`
x_plot = np.linspace(-2, 2, 1000)
y_plot = f(x_plot)

plt.figure(figsize=(12, 6))

# 轨迹图
plt.subplot(1, 2, 1)
plt.plot(x_plot, y_plot, 'b-', label='f(x)', linewidth=2)
for i, x_val in enumerate(path_beta_1):
    plt.plot(x_val, f(x_val), 'o', color=colors[i], label=f'β=1 iter {i}' if i == 0 else "")
for i, x_val in enumerate(path_beta_5):
    plt.plot(x_val, f(x_val), 's', color=colors[i], label=f'β=5 iter {i}' if i == 0 else "")
plt.title("Damped Newton Iterations")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True, alpha=0.3)
plt.legend()

# 收敛曲线
plt.subplot(1, 2, 2)
plt.plot(range(len(path_beta_1)), [f(x) for x in path_beta_1], 'ro-', label='β=1')
plt.plot(range(len(path_beta_5)), [f(x) for x in path_beta_5], 'bs-', label='β=5')
plt.title("Function Value Convergence")
plt.xlabel("Iteration")
plt.ylabel("f(x)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.suptitle("Effect of Damping in Newton's Method", fontsize=16, y=1.05)
plt.show()

# `打印最终结果`
print(f"β=1.0 最终结果: x = {path_beta_1[-1]:.6f}, f(x) = {f(path_beta_1[-1]):.6f}")
print(f"β=5.0 最终结果: x = {path_beta_5[-1]:.6f}, f(x) = {f(path_beta_5[-1]):.6f}")