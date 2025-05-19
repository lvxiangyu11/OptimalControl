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
    cost = 0.5 * xhist[:, -1].T @ Qn @ xhist[:, -1]
    for k in range(N-1):
        cost += 0.5 * xhist[:, k].T @ Q @ xhist[:, k] + 0.5 * uhist[k].T * R * uhist[k]
    return cost

def rollout(x0, uhist):
    xhist = np.zeros((n, N))
    xhist[:, 0] = x0.flatten()
    for k in range(N-1):
        xhist[:, k+1] = A @ xhist[:, k] + B.flatten() * uhist[k]
    return xhist

xhist = np.zeros((n, N))
xhist[:, 0] = x0.flatten()
uhist = np.zeros(N-1)
Delta_u = np.ones(N-1)
lambda_hist = np.zeros((n, N))

xhist = rollout(x0, uhist)
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

while np.max(np.abs(Delta_u)) > 1e-2:
    lambda_hist[:, N-1] = Qn @ xhist[:, N-1]
    for k in range(N-2, -1, -1):
        Delta_u[k] = -(uhist[k] + (1/R) * B.T @ lambda_hist[:, k+1])
        lambda_hist[:, k] = Q @ xhist[:, k] + A.T @ lambda_hist[:, k+1]

    alpha = 1.0
    unew = uhist + alpha * Delta_u
    xnew = rollout(x0, unew)
    while J(xnew, unew) > J(xhist, uhist) - b*alpha*np.dot(Delta_u, Delta_u):
        alpha = 0.5 * alpha
        unew = uhist + alpha * Delta_u
        xnew = rollout(x0, unew)

    uhist = unew
    xhist = xnew
    iter_count += 1

    frame_files.append(save_frame(xhist, uhist, iter_count))

    if iter_count % 10 == 0:
        print(f"迭代 {iter_count}, 最大梯度: {np.max(np.abs(Delta_u))}, 代价: {J(xhist, uhist)}")
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