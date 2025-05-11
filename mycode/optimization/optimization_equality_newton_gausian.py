import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from autograd import jacobian
import autograd.numpy as anp

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False   # 正确显示负号

# 定义函数
def f(x):
    Q = np.diag([0.5, 1])
    x_minus = x - np.array([1, 0])
    return 0.5 * x_minus.dot(Q).dot(x_minus)

def grad_f(x):
    Q = np.diag([0.5, 1])
    return Q.dot(x - np.array([1, 0]))

def hessian_f(x):
    return np.diag([0.5, 1])

def c(x):
    return x[0]**2 + 2*x[0] - x[1]

def grad_c(x):
    return np.array([2*x[0]+2, -1])

def lambda_grad_c(x, lambda0):
    # 确保 x 是浮动类型
    x = anp.array(x, dtype=anp.float64)  # 将 x 转换为 float64 类型
    return lambda0 * anp.array([2*x[0]+2, -1], dtype=anp.float64)


def plot_landscape():
    Nsamp = 20
    x = np.linspace(-4, 4, Nsamp)
    y = np.linspace(-4, 4, Nsamp)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((Nsamp, Nsamp))
    
    for j in range(Nsamp):
        for k in range(Nsamp):
            Z[j, k] = f(np.array([X[j, k], Y[j, k]]))
    
    plt.contour(X, Y, Z)
    xc = np.linspace(-3.2, 1.2, Nsamp)
    yc = xc**2 + 2.0*xc
    plt.plot(xc, yc, 'y', label='约束条件')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('优化问题的等高线和约束')


def newton_step(x0, lambda0):
    x0 = anp.array(x0, dtype=anp.float64)
    lambda0 = float(lambda0)
    
    H = hessian_f(x0)
    C = grad_c(x0).reshape(1, -1)
    
    lambda_grad_c_jacobian = jacobian(lambda x: lambda_grad_c(x, lambda0))
    jacobian_term = lambda_grad_c_jacobian(x0)
    
    H_adjusted = H + jacobian_term # 这里如果选择初值为[-3, 2]将导致不收敛
    KKT_matrix = np.block([
        [H_adjusted, C.T],
        [C, np.array([[0.]])]
    ])
    
    grad_term = -grad_f(x0) - C.T.flatten() * lambda0
    constraint_term = -np.array([c(x0)])
    rhs = np.concatenate([grad_term, constraint_term])
    
    delta_z = np.linalg.solve(KKT_matrix, rhs)
    
    delta_x = delta_z[:2]
    delta_lambda = delta_z[2]
    
    return x0 + delta_x, lambda0 + delta_lambda

def gauss_newton_step(x0, lambda0):
    # Gauss-Newton方法实现 - 忽略约束曲率（二阶导数）
    # 确保x0是一维数组
    x0 = np.ravel(x0)
    lambda0 = np.array([lambda0]).ravel()
    
    # 计算目标函数的Hessian矩阵 (仅使用一阶项)
    H = hessian_f(x0)
    
    # 计算约束的雅可比矩阵
    C = grad_c(x0).reshape(1, -1)
    
    # Gauss-Newton简化：忽略约束的二阶导数影响
    # 直接使用目标函数的Hessian而不添加约束的二阶导数影响
    
    # 构建简化的KKT矩阵
    KKT_matrix = np.block([
        [H, C.T],
        [C, np.array([[0]])]
    ])
    
    # 构建右侧向量
    rhs = np.concatenate([
        -grad_f(x0) - C.T @ lambda0,
        -np.array([c(x0)])
    ])
    
    # 求解线性方程组
    delta_z = np.linalg.solve(KKT_matrix, rhs)
    
    delta_x = delta_z[:2]
    delta_lambda = delta_z[2]
    
    return x0 + delta_x, lambda0[0] + delta_lambda

# 初始化猜测点
x_guess = np.array([-1, -1])
lambda_guess = 0.0

def animate_newton():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制初始景观
    Nsamp = 20
    x = np.linspace(-4, 4, Nsamp)
    y = np.linspace(-4, 4, Nsamp)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((Nsamp, Nsamp))
    
    for j in range(Nsamp):
        for k in range(Nsamp):
            Z[j, k] = f(np.array([X[j, k], Y[j, k]]))
    
    # 初始点
    x_history = [x_guess]
    lambda_history = [lambda_guess]
    
    # 第一次迭代
    x_new, lambda_new = newton_step(x_history[-1], lambda_history[-1])
    x_history.append(x_new)
    lambda_history.append(lambda_new)
    
    # 绘制所有迭代点
    iterations = 10  # 最大迭代次数
    
    for i in range(iterations):
        # 更新点
        if i > 0:  # 第一次迭代已经在上面完成
            try:
                x_new, lambda_new = newton_step(x_history[-1], lambda_history[-1])
                x_history.append(x_new)
                lambda_history.append(lambda_new)
            except np.linalg.LinAlgError:
                print(f"迭代 {i} 中出现线性代数错误，停止迭代")
                break
                
        # 清空之前的点
        ax.clear()
        
        # 重新绘制等高线和约束
        ax.contour(X, Y, Z)
        xc = np.linspace(-3.2, 1.2, Nsamp)
        yc = xc**2 + 2.0*xc
        ax.plot(xc, yc, 'y', label='约束条件')
        
        # 绘制迭代历史
        x_path = np.array(x_history)
        ax.plot(x_path[:, 0], x_path[:, 1], 'r-', marker='x')
        
        # 标记当前点
        ax.plot(x_history[-1][0], x_history[-1][1], 'ro', markersize=10)
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(f'完整牛顿法迭代 {i+1}')
        ax.legend()
        
        plt.tight_layout()
        plt.pause(0.1)
        
        # 检查收敛性
        if i > 0 and np.linalg.norm(np.array(x_history[-1]) - np.array(x_history[-2])) < 1e-5:
            print(f"在迭代 {i+1} 后收敛，请自行关闭")
            break
    
    plt.show()
    
    # 打印最终结果
    print(f"最终点: x = {x_history[-1]}")
    print(f"最终拉格朗日乘子: λ = {lambda_history[-1]}")
    
    # 计算最终点的海森矩阵
    final_x = x_history[-1]
    final_lambda = lambda_history[-1]
    H = hessian_f(final_x) + final_lambda * np.array([[2, 0], [0, 0]])  # 包含约束二阶导数的影响
    print(f"最终海森矩阵:\n{H}")

def animate_gauss_newton():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制初始景观
    Nsamp = 20
    x = np.linspace(-4, 4, Nsamp)
    y = np.linspace(-4, 4, Nsamp)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((Nsamp, Nsamp))
    
    for j in range(Nsamp):
        for k in range(Nsamp):
            Z[j, k] = f(np.array([X[j, k], Y[j, k]]))
    
    # 初始点
    x_history = [x_guess]
    lambda_history = [lambda_guess]
    
    # 第一次迭代
    x_new, lambda_new = gauss_newton_step(x_history[-1], lambda_history[-1])
    x_history.append(x_new)
    lambda_history.append(lambda_new)
    
    # 绘制所有迭代点
    iterations = 10  # 最大迭代次数
    
    for i in range(iterations):
        # 更新点
        if i > 0:  # 第一次迭代已经在上面完成
            try:
                x_new, lambda_new = gauss_newton_step(x_history[-1], lambda_history[-1])
                x_history.append(x_new)
                lambda_history.append(lambda_new)
            except np.linalg.LinAlgError:
                print(f"迭代 {i} 中出现线性代数错误，停止迭代")
                break
                
        # 清空之前的点
        ax.clear()
        
        # 重新绘制等高线和约束
        ax.contour(X, Y, Z)
        xc = np.linspace(-3.2, 1.2, Nsamp)
        yc = xc**2 + 2.0*xc
        ax.plot(xc, yc, 'y', label='约束条件')
        
        # 绘制迭代历史
        x_path = np.array(x_history)
        ax.plot(x_path[:, 0], x_path[:, 1], 'r-', marker='x')
        
        # 标记当前点
        ax.plot(x_history[-1][0], x_history[-1][1], 'ro', markersize=10)
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(f'高斯牛顿法迭代 {i+1}')
        ax.legend()
        
        plt.tight_layout()
        plt.pause(0.1)
        
        # 检查收敛性
        if i > 0 and np.linalg.norm(np.array(x_history[-1]) - np.array(x_history[-2])) < 1e-5:
            print(f"在迭代 {i+1} 后收敛，请自行关闭")
            break
    
    plt.show()
    
    # 打印最终结果
    print(f"最终点: x = {x_history[-1]}")
    print(f"最终拉格朗日乘子: λ = {lambda_history[-1]}")
    
    # 计算最终点的海森矩阵（仅使用一阶项）
    final_x = x_history[-1]
    final_lambda = lambda_history[-1]
    H = hessian_f(final_x)  # Gauss-Newton法忽略约束的二阶导数影响
    print(f"最终海森矩阵(仅包含目标函数Hessian):\n{H}")

def compare_methods():
    """比较牛顿法和高斯牛顿法的收敛速度与计算效率"""
    # 初始点
    lambda_start = 0.0
    
    # 存储两种方法的历史记录
    x_newton = [np.array([-1,-1])]
    lambda_newton = [lambda_start]
    x_gauss = [np.array([-3, 2]).copy()]
    lambda_gauss = [lambda_start]
    
    # 迭代直到收敛
    max_iter = 20
    newton_converged = False
    gauss_converged = False
    newton_iters = 0
    gauss_iters = 0
    
    for i in range(max_iter):
        # 牛顿法迭代
        if not newton_converged:
            try:
                x_new, lambda_new = newton_step(x_newton[-1], lambda_newton[-1])
                x_newton.append(x_new)
                lambda_newton.append(lambda_new)
                newton_iters += 1
                
                # 检查收敛性
                if np.linalg.norm(np.array(x_newton[-1]) - np.array(x_newton[-2])) < 1e-5:
                    newton_converged = True
            except np.linalg.LinAlgError:
                print("牛顿法出现线性代数错误")
                break
        
        # 高斯牛顿法迭代
        if not gauss_converged:
            try:
                x_new, lambda_new = gauss_newton_step(x_gauss[-1], lambda_gauss[-1])
                x_gauss.append(x_new)
                lambda_gauss.append(lambda_new)
                gauss_iters += 1
                
                # 检查收敛性
                if np.linalg.norm(np.array(x_gauss[-1]) - np.array(x_gauss[-2])) < 1e-5:
                    gauss_converged = True
            except np.linalg.LinAlgError:
                print("高斯牛顿法出现线性代数错误")
                break
        
        # 如果两种方法都已收敛，则停止迭代
        if newton_converged and gauss_converged:
            break
    
    # 绘制对比图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制等高线和约束
    Nsamp = 20
    x = np.linspace(-4, 4, Nsamp)
    y = np.linspace(-4, 4, Nsamp)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((Nsamp, Nsamp))
    
    for j in range(Nsamp):
        for k in range(Nsamp):
            Z[j, k] = f(np.array([X[j, k], Y[j, k]]))
    
    ax.contour(X, Y, Z)
    xc = np.linspace(-3.2, 1.2, Nsamp)
    yc = xc**2 + 2.0*xc
    ax.plot(xc, yc, 'y', label='约束条件')
    
    # 绘制牛顿法路径
    x_path = np.array(x_newton)
    ax.plot(x_path[:, 0], x_path[:, 1], 'r-', marker='x', label=f'完整牛顿法 (迭代{newton_iters}次)')
    
    # 绘制高斯牛顿法路径
    x_path = np.array(x_gauss)
    ax.plot(x_path[:, 0], x_path[:, 1], 'b-', marker='o', label=f'高斯牛顿法 (迭代{gauss_iters}次)')
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('牛顿法与高斯牛顿法收敛比较')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 打印结果对比
    print(f"完整牛顿法：")
    print(f"  迭代次数: {newton_iters}")
    print(f"  最终点: x = {x_newton[-1]}")
    print(f"  最终拉格朗日乘子: λ = {lambda_newton[-1]}")
    
    print(f"\n高斯牛顿法：")
    print(f"  迭代次数: {gauss_iters}")
    print(f"  最终点: x = {x_gauss[-1]}")
    print(f"  最终拉格朗日乘子: λ = {lambda_gauss[-1]}")
    
    # 计算每步的残差变化
    newton_residuals = []
    gauss_residuals = []
    
    for i in range(len(x_newton)-1):
        newton_residuals.append(np.linalg.norm(np.array(x_newton[i+1]) - np.array(x_newton[i])))
    
    for i in range(len(x_gauss)-1):
        gauss_residuals.append(np.linalg.norm(np.array(x_gauss[i+1]) - np.array(x_gauss[i])))
    
    # 绘制残差变化
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(newton_residuals)+1), newton_residuals, 'r-', marker='x', label='完整牛顿法')
    plt.semilogy(range(1, len(gauss_residuals)+1), gauss_residuals, 'b-', marker='o', label='高斯牛顿法')
    plt.xlabel('迭代次数')
    plt.ylabel('残差 (对数刻度)')
    plt.title('收敛速度对比')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# 用户可以选择运行以下任意一个函数：
# animate_newton()        # 运行完整牛顿法动画
# animate_gauss_newton()  # 运行高斯牛顿法动画
compare_methods()       # 运行两种方法的完整比较