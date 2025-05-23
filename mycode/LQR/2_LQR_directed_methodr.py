import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import shutil
import os
from numpy import linalg as LA

plt.rcParams['font.sans-serif'] = ['SimHei']   
plt.rcParams['axes.unicode_minus'] = False     

h = 0.1
A = np.array([[1, h], [0, 1]])
B = np.array([[0.5*h*h], [h]])
n = 2
m = 1
Tfinal = 5.0
N = int(Tfinal/h) + 1
thist = np.arange(0, h*N, h)

x0 = np.array([[1.0], [0]])

Q = np.eye(2)
R = 0.1
Qn = np.eye(2)

def J(xhist, uhist):
    """
    计算系统的代价函数值
    
    参数:
        xhist: 形状为(n, N)的数组，存储状态轨迹
        uhist: 形状为(N-1)的数组，存储控制输入序列
    
    返回:
        cost: 总代价函数值
    """
    # 计算终端状态代价
    cost = 0.5 * xhist[:, -1].T @ Qn @ xhist[:, -1]
    
    # 累加所有时间步的状态代价和控制代价
    for k in range(N-1):
        # 状态代价: 0.5 * x^T * Q * x
        # 控制代价: 0.5 * u^T * R * u
        cost += 0.5 * xhist[:, k].T @ Q @ xhist[:, k] + 0.5 * uhist[k] * R * uhist[k]
    return cost

def rollout(x0, uhist):
    """
    根据初始状态和控制序列前向模拟系统
    
    参数:
        x0: 形状为(n, 1)的数组，表示初始状态
        uhist: 形状为(N-1)的数组，表示控制输入序列
    
    返回:
        xhist: 形状为(n, N)的数组，表示系统状态轨迹
    """
    # 初始化状态历史数组
    xhist = np.zeros((n, N))
    # 设置初始状态
    xhist[:, 0] = x0.flatten()
    
    # 前向模拟系统动态方程 x_{k+1} = A*x_k + B*u_k
    for k in range(N-1):
        xhist[:, k+1] = A @ xhist[:, k] + B.flatten() * uhist[k]
    return xhist

def rollout_rk4(x0, uhist):
    """
    使用RK4方法根据初始状态和控制序列前向模拟系统
    
    参数:
        x0: 形状为(n, 1)的数组，表示初始状态
        uhist: 形状为(N-1)的数组，表示控制输入序列
    
    返回:
        xhist: 形状为(n, N)的数组，表示系统状态轨迹
    """
    # 初始化状态历史数组
    xhist = np.zeros((n, N))
    # 设置初始状态
    xhist[:, 0] = x0.flatten()
    
    # 定义系统动态方程 dx/dt = f(t, x, u)
    def f(t, x, u):
        # 转换离散线性系统 x_{k+1} = Ax_k + Bu_k 为连续时间版本
        # 连续时间系统可表示为 dx/dt = (A-I)/h*x + B/h*u
        # 其中A和B为原离散系统的系数矩阵
        # 注: 这种转换是粗略的，实际需要根据系统原始建模方式调整
        A_continuous = (A - np.eye(n)) / h
        B_continuous = B / h
        return A_continuous @ x + B_continuous.flatten() * u
    
    # 前向模拟系统使用RK4方法
    for k in range(N-1):
        x = xhist[:, k]
        u = uhist[k]
        t = thist[k]
        
        # RK4方法四个阶段
        k1 = f(t, x, u)
        k2 = f(t + h/2, x + h/2 * k1, u)
        k3 = f(t + h/2, x + h/2 * k2, u)
        k4 = f(t + h, x + h * k3, u)
        
        # 计算下一个状态
        xhist[:, k+1] = x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return xhist

xhist = np.zeros((n, N))
xhist[:, 0] = x0.flatten()
uhist = np.zeros(N-1)
Delta_u = np.ones(N-1)
lambda_hist = np.zeros((n, N))

# xhist = rollout(x0, uhist)
xhist = rollout_rk4(x0, uhist)
initial_cost = J(xhist, uhist)
print(f"初始代价: {initial_cost}")

b = 1e-2
alpha = 1.0
iter_count = 0

# 创建一个临时目录存放帧
frame_dir = "frames_gif"
if not os.path.exists(frame_dir):
    os.makedirs(frame_dir)
images = []

def save_frame(xhist, uhist, iter_count):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(thist, xhist[0, :], 'b-', label='位置')
    plt.plot(thist, xhist[1, :], 'r-', label='速度')
    plt.xlabel('时间')
    plt.ylabel('状态')
    plt.legend()
    plt.title(f'状态轨迹 - 第{iter_count}次迭代')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(thist[:-1], uhist, 'g-', label='控制输入')
    plt.xlabel('时间')
    plt.ylabel('控制')
    plt.legend()
    plt.title('控制轨迹')
    plt.grid(True)
    plt.tight_layout()
    fname = f"{frame_dir}/frame_{iter_count:04d}.png"
    plt.savefig(fname)
    plt.close()
    return fname

# 保存初始帧
frame_files = [save_frame(xhist, uhist, iter_count)]

# 迭代优化循环，直到控制变化量小于阈值
while np.max(np.abs(Delta_u)) > 1e-2:
    # 设置终端状态的协状态值（lambda）为终端代价的梯度
    lambda_hist[:, N-1] = Qn @ xhist[:, N-1]
    
    # 从终端时刻向后迭代计算协状态和控制更新量
    for k in range(N-2, -1, -1):
        # 计算控制梯度方向 Delta_u = -∇J = -(u + R^-1 * B^T * lambda)
        Delta_u[k] = -(uhist[k] + (1/R) * B.T @ lambda_hist[:, k+1])
        # 更新协状态λ，从终端向初始状态反向传播
        # lambda_k = Q*x_k + A^T * lambda_{k+1}
        lambda_hist[:, k] = Q @ xhist[:, k] + A.T @ lambda_hist[:, k+1]

    # 线搜索确定步长 alpha
    alpha = 1.0  # 初始步长
    # 计算新的控制序列和对应状态轨迹
    unew = uhist + alpha * Delta_u
    # xnew = rollout(x0, unew)
    xnew = rollout_rk4(x0, unew)
    
    # Armijo线搜索条件: J(x_new, u_new) <= J(x, u) - b*alpha*||Delta_u||^2
    # 如果不满足条件，减小步长并重新尝试
    while J(xnew, unew) > J(xhist, uhist) - b*alpha*np.dot(Delta_u, Delta_u):
        alpha = 0.5 * alpha  # 步长减半
        unew = uhist + alpha * Delta_u
        # xnew = rollout(x0, unew)
        xnew = rollout_rk4(x0, unew)

    # 更新控制和状态历史
    uhist = unew
    xhist = xnew
    iter_count += 1

    # 保存当前迭代的动画帧
    frame_files.append(save_frame(xhist, uhist, iter_count))

    # 每10次迭代输出一次进度信息
    if iter_count % 10 == 0:
        print(f"迭代 {iter_count}, 最大梯度: {np.max(np.abs(Delta_u))}, 代价: {J(xhist, uhist)}")
    
    # 设置最大迭代次数限制
    if iter_count > 100:
        print("达到最大迭代次数")
        break

print(f"总迭代次数: {iter_count}")
print(f"最终代价: {J(xhist, uhist)}")

# 生成GIF
images = [imageio.imread(fname) for fname in frame_files]
imageio.mimsave('simulation_optimization.gif', images, duration=0.3)

# 删除临时帧文件和文件夹
try:
    shutil.rmtree(frame_dir)  # 递归删除文件夹及内容
except Exception as e:
    print(f"删除临时文件夹失败: {e}")

print("GIF已保存为 simulation_optimization.gif")