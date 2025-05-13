import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from matplotlib.font_manager import FontProperties

# 设置中文字体支持
try:
    # 尝试设置微软雅黑字体
    font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
except:
    try:
        # 尝试设置思源黑体字体
        font = FontProperties(fname=r"/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")
    except:
        # 如无法找到中文字体，使用默认字体
        font = FontProperties()

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置对角矩阵 Q
Q = np.diag([0.5, 1.0])

# 定义目标函数
def f(x):
    """计算目标函数值"""
    diff = x - np.array([1.0, 0.0])
    return 0.5 * diff.T @ Q @ diff

# 定义目标函数梯度
def grad_f(x):
    """计算目标函数梯度"""
    return Q @ (x - np.array([1.0, 0.0]))

# 定义目标函数的Hessian矩阵
def hessian_f(x):
    """计算目标函数的Hessian矩阵"""
    return Q

# 定义约束条件
A = np.array([1.0, -1.0])
b = -1.0

def c(x):
    """计算约束条件值"""
    return np.dot(A, x) - b

def grad_c(x):
    """计算约束条件梯度"""
    return A

# 绘制函数轮廓和约束条件
def plot_landscape(xguess=None, current_point=None):
    """绘制目标函数等高线和约束条件"""
    plt.clf()  # 清除当前图形
    
    # 创建网格
    Nsamp = 20
    x = np.linspace(-4, 4, Nsamp)
    y = np.linspace(-4, 4, Nsamp)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((Nsamp, Nsamp))
    
    # 计算目标函数值
    for i in range(Nsamp):
        for j in range(Nsamp):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    
    # 绘制等高线
    contour = plt.contour(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, label='目标函数值')
    
    # 绘制约束条件边界
    xc = np.linspace(-4, 3, Nsamp)
    yc = xc + 1  # 对应约束条件 x1 - x2 = 1
    plt.plot(xc, yc, 'y-', label='约束条件：x₁ - x₂ = 1')
    
    # 如果有迭代点，绘制迭代路径
    if xguess is not None and xguess.shape[1] > 1:
        plt.plot(xguess[0], xguess[1], 'r-', linewidth=1)
        plt.plot(xguess[0, :-1], xguess[1, :-1], 'rx', markersize=8)
    
    # 如果有当前点，特别标记
    if current_point is not None:
        plt.plot(current_point[0], current_point[1], 'ro', markersize=10)
    
    plt.grid(True)
    plt.xlabel('x₁', fontproperties=font)
    plt.ylabel('x₂', fontproperties=font)
    plt.title('优化问题的可视化', fontproperties=font)
    plt.legend(prop=font)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

# 增广拉格朗日函数
def La(x, lam, rho):
    """计算增广拉格朗日函数值"""
    p = max(0, c(x))
    return f(x) + lam * p + (rho/2) * (p**2)

# 牛顿法求解
def newton_solve(x0, lam, rho, tol=1e-8):
    """使用牛顿法求解增广拉格朗日函数的最小值点"""
    x = x0.copy()
    p = max(0, c(x))
    
    # 初始化约束梯度矩阵
    C = np.zeros((1, 2))
    if c(x) >= 0:
        C = grad_c(x).reshape(1, 2)
    
    # 计算目标函数梯度
    g = grad_f(x) + (lam + rho * p) * C.T.flatten()
    
    # 牛顿法迭代
    iter_count = 0
    max_iter = 100
    
    while np.linalg.norm(g) >= tol and iter_count < max_iter:
        # 计算Hessian矩阵
        H = hessian_f(x) + rho * C.T @ C
        
        # 计算牛顿方向
        delta_x = np.linalg.solve(H, -g)
        
        # 更新x
        x = x + delta_x
        
        # 更新约束相关量
        p = max(0, c(x))
        C = np.zeros((1, 2))
        if c(x) >= 0:
            C = grad_c(x).reshape(1, 2)
        
        # 更新梯度
        g = grad_f(x) + (lam + rho * p) * C.T.flatten()
        iter_count += 1
    
    return x

# 主程序
def main():
    # 初始猜测
    xguess = np.array([[-3.0], [2.0]])
    lambda_guess = np.array([0.0])
    rho = 1.0
    
    # 创建动态图形
    plt.figure(figsize=(10, 8))
    plt.ion()  # 开启交互模式
    
    # 绘制初始状态
    plot_landscape(xguess, xguess[:, 0])
    plt.draw()
    plt.pause(0.5)  # 暂停0.5秒
    
    # 定义收敛条件
    tol = 1e-8
    max_iter = 30
    iter_count = 0
    converged = False
    
    while not converged and iter_count < max_iter:
        # 当前点
        current_x = xguess[:, -1]
        
        # 牛顿法迭代
        xnew = newton_solve(current_x, lambda_guess[-1], rho)
        lambda_new = max(0, lambda_guess[-1] + rho * c(xnew))
        
        # 更新猜测
        xguess = np.hstack((xguess, xnew.reshape(-1, 1)))
        lambda_guess = np.append(lambda_guess, lambda_new)
        
        # 增加惩罚因子
        rho = 10.0 * rho
        
        # 检查收敛性 - 如果连续两次迭代的结果非常接近，认为已收敛
        if iter_count > 0:
            x_diff = np.linalg.norm(xguess[:, -1] - xguess[:, -2])
            lambda_diff = abs(lambda_guess[-1] - lambda_guess[-2])
            if x_diff < tol and lambda_diff < tol:
                converged = True
        
        # 绘制当前状态
        plot_landscape(xguess, xguess[:, -1])
        plt.draw()
        plt.pause(0.5)  # 暂停0.5秒
        
        iter_count += 1
        
        if converged:
            print("优化已收敛!")
            break
    
    # 最终结果绘图
    plot_landscape(xguess, xguess[:, -1])
    plt.ioff()  # 关闭交互模式
    
    # 添加额外的文本说明最终结果
    final_x = xguess[:, -1]
    plt.annotate(f'最终结果: ({final_x[0]:.4f}, {final_x[1]:.4f})',
                xy=(final_x[0], final_x[1]), xytext=(final_x[0]-1, final_x[1]-1),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontproperties=font, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.title('优化最终结果', fontproperties=font)
    plt.show()
    
    print("\n最终结果:")
    print(f"x1 = {xguess[0, -1]:.6f}")
    print(f"x2 = {xguess[1, -1]:.6f}")
    print(f"λ = {lambda_guess[-1]:.6f}")
    print(f"约束条件值: {c(xguess[:, -1]):.6e}")
    print(f"目标函数值: {f(xguess[:, -1]):.6f}")

if __name__ == "__main__":
    main()