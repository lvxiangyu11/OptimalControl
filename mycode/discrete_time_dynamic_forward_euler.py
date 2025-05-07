import numpy as np
from src.PendulumVisualizer import *


def pendulum_dynamics(x):
    l = 1.0
    g = 9.81
    
    theta = x[0]
    theta_dot = x[1]
    theta_ddot = -(g/l) * np.sin(theta)
    
    return np.array([theta_dot, theta_ddot])

def pendulum_forward_euler(fun, x0, Tf, h):
    t = np.arange(0, Tf + h, h)
    
    x_hist = np.zeros((len(x0), len(t)))
    x_hist[:, 0] = x0
    
    for k in range(len(t) - 1):
        x_hist[:, k+1] = x_hist[:, k] + h * fun(x_hist[:, k])
    
    return x_hist, t

# 使用示例
if __name__ == "__main__":
    # 初始条件和参数
    x0 = np.array([0.1, 0])  # 初始角度和角速度
    Tf = 10.0  # 模拟总时间(秒)
    h = 0.01  # 时间步长
    
    # 计算单摆运动
    x_hist, t_hist = pendulum_forward_euler(pendulum_dynamics, x0, Tf, h)
    
    # 创建可视化对象并显示动画
    visualizer = PendulumVisualizer(pendulum_length=1.0)
    visualizer.visualize(x_hist, t_hist)