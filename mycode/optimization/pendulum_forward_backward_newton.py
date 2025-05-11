import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as anp
from autograd import jacobian
from src.PendulumVisualizer import *

def pendulum_dynamics(x):
    l = 1.0
    g = 9.81
    
    theta = x[0]
    theta_dot = x[1]
    
    theta_ddot = -(g/l)*np.sin(theta)
    
    return np.array([theta_dot, theta_ddot])

# 为 autograd 创建一个特殊版本的动力学函数
def pendulum_dynamics_autograd(x):
    l = 1.0
    g = 9.81
    
    theta = x[0]
    theta_dot = x[1]
    
    theta_ddot = -(g/l)*anp.sin(theta)
    
    return anp.array([theta_dot, theta_ddot])

def backward_euler_step_fixed_point(fun, x0, h):
    xn = x0.copy()
    e = [np.linalg.norm(x0 + h*fun(xn) - xn)]
    
    while e[-1] > 1e-8:
        xn = x0 + h*fun(xn)
        e.append(np.linalg.norm(x0 + h*fun(xn) - xn))
    
    return xn, np.array(e)

def backward_euler_step_newton(fun, x0, h):
    xn = x0.copy()
    
    # 定义残差函数
    def residual(x):
        return x0 + h*fun(x) - x
    
    # 为autograd定义残差函数
    def residual_autograd(x):
        return anp.array(x0) + h*pendulum_dynamics_autograd(x) - anp.array(x)
    
    # 使用autograd计算雅可比矩阵
    res_jac = jacobian(residual_autograd)
    
    r = residual(xn)
    e = [np.linalg.norm(r)]
    
    while e[-1] > 1e-8:
        # 获取当前点的雅可比矩阵
        dr = res_jac(xn)
        
        # 解线性系统
        delta = np.linalg.solve(dr, r)
        xn = xn - delta
        
        r = residual(xn)
        e.append(np.linalg.norm(r))
    
    return xn, np.array(e)

def backward_euler_fixed_point(fun, x0, Tf, h):
    t = np.arange(0, Tf+h/2, h)
    
    x_hist = np.zeros((len(x0), len(t)))
    x_hist[:,0] = x0
    
    for k in range(len(t)-1):
        x_hist[:,k+1], e = backward_euler_step_fixed_point(fun, x_hist[:,k], h)
    
    return x_hist, t

def backward_euler_newton(fun, x0, Tf, h):
    t = np.arange(0, Tf+h/2, h)
    
    x_hist = np.zeros((len(x0), len(t)))
    x_hist[:,0] = x0
    
    for k in range(len(t)-1):
        x_hist[:,k+1], e = backward_euler_step_newton(fun, x_hist[:,k], h)
    
    return x_hist, t

# 初始条件和模拟
x0 = np.array([0.1, 0])

# 运行两种不同的方法
x_hist1, t_hist1 = backward_euler_fixed_point(pendulum_dynamics, x0, 10, 0.01)
x_hist2, t_hist2 = backward_euler_newton(pendulum_dynamics, x0, 10, 0.01)

# 基础分析可视化
plt.figure(figsize=(12, 10))

# 绘制两种方法的钟摆角度随时间变化
plt.subplot(2, 2, 1)
plt.plot(t_hist1, x_hist1[0,:], label='定点迭代法')
plt.plot(t_hist2, x_hist2[0,:], '--', label='牛顿法')
plt.xlabel('时间 (秒)')
plt.ylabel('角度 (弧度)')
plt.title('钟摆角度随时间变化')
plt.legend()

# 绘制两种方法之间的差异
plt.subplot(2, 2, 2)
diff = np.abs(x_hist1 - x_hist2)
plt.plot(t_hist1, diff[0,:])
plt.xlabel('时间 (秒)')
plt.ylabel('角度差异 (弧度)')
plt.title(f'方法差异 (最大值: {np.max(diff):.2e})')

# 计算并绘制单步收敛性
xn_fixed, e1 = backward_euler_step_fixed_point(pendulum_dynamics, x0, 0.1)
xn_newton, e2 = backward_euler_step_newton(pendulum_dynamics, x0, 0.1)

plt.subplot(2, 2, 3)
plt.semilogy(e1, 'o-')
plt.xlabel('迭代次数')
plt.ylabel('误差')
plt.title('定点迭代法收敛性')

plt.subplot(2, 2, 4)
plt.semilogy(e2, 'o-')
plt.xlabel('迭代次数')
plt.ylabel('误差')
plt.title('牛顿法收敛性')

plt.tight_layout()
plt.show()

# 使用PendulumVisualizer进行动画可视化
print("正在创建定点迭代法的单摆动画...")
viz1 = PendulumVisualizer()
viz1.visualize(x_hist1, t_hist1)

print("正在创建牛顿法的单摆动画...")
viz2 = PendulumVisualizer()
viz2.visualize(x_hist2, t_hist2)