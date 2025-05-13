import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize
import matplotlib as mpl

# 设置中文环境
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
mpl.rcParams['font.size'] = 12

# 定义目标函数和约束函数
def f(x):
    Q = np.diag([0.5, 1])
    target = np.array([1, 0])
    return 0.5 * np.dot((x - target).T, np.dot(Q, (x - target)))

def grad_f(x):
    Q = np.diag([0.5, 1])
    target = np.array([1, 0])
    return np.dot(Q, (x - target))

def hess_f(x):
    return np.diag([0.5, 1])

def c(x):
    return x[0]**2 + 2*x[0] - x[1]

def grad_c(x):
    return np.array([2*x[0] + 2, -1])

def jacobian_c(x):
    return grad_c(x).reshape(1, -1)

# 牛顿步骤函数
def newton_step(x, lam):
    H = hess_f(x) + lam * np.array([[2, 0], [0, 0]])
    C = jacobian_c(x)
    
    # 构建KKT矩阵
    K = np.block([
        [H, C.T],
        [C, np.zeros((1, 1))]
    ])
    
    # 构建右侧向量
    rhs = np.concatenate([-grad_f(x) - C.T @ lam, -np.array([c(x)])])
    
    # 求解线性系统
    delta_z = np.linalg.solve(K, rhs)
    
    delta_x = delta_z[:2]
    delta_lam = delta_z[2:]
    
    return delta_x, delta_lam

# 正则化牛顿步骤函数
def regularized_newton_step(x, lam):
    beta = 1.0
    H = hess_f(x) + lam * np.array([[2, 0], [0, 0]])
    C = jacobian_c(x)
    
    # 构建KKT矩阵
    K = np.block([
        [H, C.T],
        [C, np.zeros((1, 1))]
    ])
    
    # 计算特征值
    e = np.linalg.eigvals(K)
    
    # 正则化矩阵直到满足鞍点条件
    while not (np.sum(e > 0) == len(x) and np.sum(e < 0) == len(lam)):
        reg_matrix = np.zeros_like(K)
        reg_matrix[:2, :2] = beta * np.eye(2)
        reg_matrix[2:, 2:] = -beta * np.eye(1)
        K = K + reg_matrix
        e = np.linalg.eigvals(K)
    
    # 求解线性系统
    rhs = np.concatenate([-grad_f(x) - C.T @ lam, -np.array([c(x)])])
    delta_z = np.linalg.solve(K, rhs)
    
    delta_x = delta_z[:2]
    delta_lam = delta_z[2:]
    
    return delta_x, delta_lam

# 绘制优化景观
def plot_landscape(ax):
    ax.clear()
    Nsamp = 20
    x = np.linspace(-4, 4, Nsamp)
    y = np.linspace(-4, 4, Nsamp)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((Nsamp, Nsamp))
    
    for j in range(Nsamp):
        for k in range(Nsamp):
            Z[j, k] = f(np.array([X[j, k], Y[j, k]]))
    
    # 绘制等高线
    cs = ax.contour(X, Y, Z)
    
    # 绘制约束曲线
    xc = np.linspace(-3.2, 1.2, Nsamp)
    yc = xc**2 + 2*xc
    ax.plot(xc, yc, 'y-', linewidth=2, label='约束: x₁²+2x₁-x₂=0')
    
    ax.set_title('优化算法可视化')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.legend(loc='upper right')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.grid(True)
    
    return cs

# 标准牛顿法可视化
def visualize_standard_newton():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 初始点和初始拉格朗日乘子
    x_guess = np.array([-3, 2]).reshape(2, 1)
    lam_guess = np.array([0.0]).reshape(1, 1)
    
    max_iter = 10
    convergence = False
    iterations = 0
    
    # 存储所有迭代点
    x_history = [x_guess.flatten()]
    
    def update(frame):
        nonlocal x_guess, lam_guess, convergence, iterations
        
        if convergence or iterations >= max_iter:
            return
        
        try:
            delta_x, delta_lam = newton_step(x_guess.flatten(), lam_guess.flatten())
            x_new = x_guess.flatten() + delta_x
            lam_new = lam_guess.flatten() + delta_lam
            
            x_guess = x_new.reshape(2, 1)
            lam_guess = lam_new.reshape(1, 1)
            
            x_history.append(x_guess.flatten())
            
            # 检查收敛
            if np.linalg.norm(delta_x) < 1e-6 and abs(c(x_guess.flatten())) < 1e-6:
                convergence = True
                
            iterations += 1
                
        except np.linalg.LinAlgError:
            print("线性代数错误 - 可能是矩阵奇异")
            convergence = True
        
        # 更新图像
        cs = plot_landscape(ax)
        x_path = np.array(x_history)
        ax.plot(x_path[:, 0], x_path[:, 1], 'r-o', markersize=8, linewidth=1.5, label='牛顿法路径')
        ax.set_title(f'标准牛顿法 (迭代 {iterations})')
        
        # 添加当前点的特殊标记
        if len(x_path) > 0:
            ax.plot(x_path[-1, 0], x_path[-1, 1], 'ro', markersize=10)
        
        # 显示文本信息
        if convergence:
            ax.text(0.05, 0.95, f"最终点: ({x_guess[0,0]:.3f}, {x_guess[1,0]:.3f})\n约束值: {c(x_guess.flatten()):.6f}", 
                   transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        else:
            ax.text(0.05, 0.95, f"当前点: ({x_guess[0,0]:.3f}, {x_guess[1,0]:.3f})\n约束值: {c(x_guess.flatten()):.6f}", 
                   transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        # 更新图例
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 1:  # 避免重复添加图例
            handles = handles[:1]
            labels = labels[:1]
        ax.legend(handles + [plt.Line2D([0], [0], color='r', marker='o', linestyle='-')], 
                 labels + ['牛顿法路径'], loc='upper right')
    
    ani = FuncAnimation(fig, update, frames=max_iter+1, interval=800, repeat=False)
    plt.tight_layout()
    plt.show()

# 正则化牛顿法可视化
def visualize_regularized_newton():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 初始点和初始拉格朗日乘子
    x_guess = np.array([-3, 2]).reshape(2, 1)
    lam_guess = np.array([0.0]).reshape(1, 1)
    
    max_iter = 10
    convergence = False
    iterations = 0
    
    # 存储所有迭代点
    x_history = [x_guess.flatten()]
    
    def update(frame):
        nonlocal x_guess, lam_guess, convergence, iterations
        
        if convergence or iterations >= max_iter:
            return
        
        try:
            delta_x, delta_lam = regularized_newton_step(x_guess.flatten(), lam_guess.flatten())
            x_new = x_guess.flatten() + delta_x
            lam_new = lam_guess.flatten() + delta_lam
            
            x_guess = x_new.reshape(2, 1)
            lam_guess = lam_new.reshape(1, 1)
            
            x_history.append(x_guess.flatten())
            
            # 检查收敛
            if np.linalg.norm(delta_x) < 1e-6 and abs(c(x_guess.flatten())) < 1e-6:
                convergence = True
                
            iterations += 1
                
        except np.linalg.LinAlgError:
            print("线性代数错误 - 可能是矩阵奇异")
            convergence = True
        
        # 更新图像
        cs = plot_landscape(ax)
        x_path = np.array(x_history)
        ax.plot(x_path[:, 0], x_path[:, 1], 'b-o', markersize=8, linewidth=1.5)
        ax.set_title(f'正则化牛顿法 (迭代 {iterations})')
        
        # 添加当前点的特殊标记
        if len(x_path) > 0:
            ax.plot(x_path[-1, 0], x_path[-1, 1], 'bo', markersize=10)
        
        # 显示文本信息
        if convergence:
            ax.text(0.05, 0.95, f"最终点: ({x_guess[0,0]:.3f}, {x_guess[1,0]:.3f})\n约束值: {c(x_guess.flatten()):.6f}", 
                   transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        else:
            ax.text(0.05, 0.95, f"当前点: ({x_guess[0,0]:.3f}, {x_guess[1,0]:.3f})\n约束值: {c(x_guess.flatten()):.6f}", 
                   transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        # 更新图例
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 1:  # 避免重复添加图例
            handles = handles[:1]
            labels = labels[:1]
        ax.legend(handles + [plt.Line2D([0], [0], color='b', marker='o', linestyle='-')], 
                 labels + ['正则化牛顿法路径'], loc='upper right')
    
    ani = FuncAnimation(fig, update, frames=max_iter+1, interval=800, repeat=False)
    plt.tight_layout()
    plt.show()

# 运行两种方法的可视化
print("运行标准牛顿法可视化...")
visualize_standard_newton()

print("运行正则化牛顿法可视化...")
visualize_regularized_newton()