import numpy as np
from src.PendulumVisualizer import *

def pendulum_dynamics(x):
    l = 1.0
    g = 9.81
    
    theta = x[0]
    theta_dot = x[1]
    theta_ddot = -(g/l) * np.sin(theta)
    
    return np.array([theta_dot, theta_ddot])

def pendulum_rk4(fun, x0, Tf, h):
    t = np.arange(0, Tf + h, h)
    
    x_hist = np.zeros((len(x0), len(t)))
    x_hist[:, 0] = x0
    
    for k in range(len(t) - 1):
        x = x_hist[:, k]
        
        # RK4步骤
        k1 = fun(x)
        k2 = fun(x + h/2 * k1)
        k3 = fun(x + h/2 * k2)
        k4 = fun(x + h * k3)
        
        # 更新状态
        x_hist[:, k+1] = x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return x_hist, t

def compute_rk4_jacobian_eigenvalues(h):
    """计算RK4方法的雅可比矩阵特征值"""
    # 线性系统 x' = Ax 的情况下，A的特征值为 lambda
    # 对于单摆在小角度近似下，线性化后的A矩阵为
    l = 1.0
    g = 9.81
    A = np.array([[0, 1], [-g/l, 0]])  # 线性化的单摆动力学
    
    # 计算A的特征值
    eigen_A = np.linalg.eigvals(A)
    print("单摆线性化系统的特征值:", eigen_A)
    
    # 计算RK4方法的放大矩阵特征值
    # 对于 x_{n+1} = R(hA)x_n，其中R(z)是RK4的稳定性函数
    # R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24
    
    # 计算不同特征值的放大因子
    amplification_factors = []
    eigenvalues = []
    
    # 对每个特征值计算放大因子
    for lam in eigen_A:
        z = h * lam
        R = 1 + z + z**2/2 + z**3/6 + z**4/24
        amplification_factors.append(np.abs(R))
        eigenvalues.append(R)
    
    print(f"步长 h = {h}下RK4方法的雅可比矩阵特征值:", eigenvalues)
    print(f"放大因子(特征值绝对值):", amplification_factors)
    
    return eigen_A, eigenvalues, amplification_factors

# 使用示例
if __name__ == "__main__":
    # 初始条件和参数
    x0 = np.array([0.1, 0])  # 初始角度和角速度
    Tf = 10.0  # 模拟总时间(秒)
    h = 0.01  # 时间步长
    
    # 计算单摆运动
    x_hist, t_hist = pendulum_rk4(pendulum_dynamics, x0, Tf, h)
    
    # 创建可视化对象并显示动画
    visualizer = PendulumVisualizer(pendulum_length=1.0)
    visualizer.visualize(x_hist, t_hist)
    
    # 计算并打印RK4方法的雅可比矩阵特征值
    print("\n==== RK4方法稳定性分析 ====")
    system_eigenvals, rk4_eigenvals, amplification = compute_rk4_jacobian_eigenvalues(h)
    
    # 可选：对不同的步长进行分析
    print("\n==== 不同步长下的稳定性分析 ====")
    step_sizes = [0.01, 0.05, 0.1, 0.5, 1.0]
    for step in step_sizes:
        print(f"\n步长 h = {step}")
        compute_rk4_jacobian_eigenvalues(step)
    
    # 可视化不同步长下RK4方法的稳定域
    plt.figure(figsize=(10, 8))
    
    # 在复平面上绘制单位圆
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', label='单位圆')
    
    # 计算并绘制RK4稳定域边界
    # RK4稳定域由 |R(z)| = 1 定义，其中 R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24
    x = np.linspace(-3, 1, 1000)
    y = np.linspace(-3, 3, 1000)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    
    R = 1 + Z + Z**2/2 + Z**3/6 + Z**4/24
    R_abs = np.abs(R)
    
    plt.contour(X, Y, R_abs, levels=[1], colors='r', linewidths=2)
    
    # 标记系统特征值对应的h*lambda点
    for i, eigenval in enumerate(system_eigenvals):
        for step in step_sizes:
            z = step * eigenval
            plt.scatter(z.real, z.imag, marker='x')
            plt.annotate(f'h={step}', (z.real, z.imag))
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title('RK4方法稳定域和单摆系统特征值')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()