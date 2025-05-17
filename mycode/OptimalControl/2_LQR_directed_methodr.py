import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

plt.rcParams['font.sans-serif'] = ['SimHei']   
plt.rcParams['axes.unicode_minus'] = False     

# 离散动力学参数
h = 0.1  # 时间步长
A = np.array([[1, h], [0, 1]])  # 状态转移矩阵
B = np.array([[0.5*h*h], [h]])  # 控制输入矩阵
n = 2  # 状态维度
m = 1  # 控制维度
Tfinal = 5.0  # 最终时间
N = int(Tfinal/h) + 1  # 时间步数
thist = np.arange(0, h*N, h)  # 时间序列

# 初始条件
x0 = np.array([[1.0], [0]])  # 初始状态：位置为1，速度为0

# 代价函数权重
Q = np.eye(2)  # 状态代价权重矩阵
R = 0.1  # 控制代价权重矩阵
Qn = np.eye(2)  # 终端状态代价权重矩阵

def J(xhist, uhist):
   """计算代价函数值"""
   cost = 0.5 * xhist[:, -1].T @ Qn @ xhist[:, -1]
   for k in range(N-1):
       cost += 0.5 * xhist[:, k].T @ Q @ xhist[:, k] + 0.5 * uhist[k].T * R * uhist[k]
   return cost

def rollout(x0, uhist):
   """根据初始状态和控制序列计算状态轨迹"""
   xhist = np.zeros((n, N))
   xhist[:, 0] = x0.flatten()
   for k in range(N-1):
       xhist[:, k+1] = A @ xhist[:, k] + B.flatten() * uhist[k]
   return xhist

# 初始猜测
xhist = np.zeros((n, N))
xhist[:, 0] = x0.flatten()
uhist = np.zeros(N-1)
Delta_u = np.ones(N-1)
lambda_hist = np.zeros((n, N))

# 初始rollout计算状态轨迹
xhist = rollout(x0, uhist)
initial_cost = J(xhist, uhist)
print(f"初始代价: {initial_cost}")

# 线搜索参数
b = 1e-2  # 线搜索容差
alpha = 1.0  # 步长
iter_count = 0  # 迭代计数

# 主迭代循环
while np.max(np.abs(Delta_u)) > 1e-2:  # 当梯度足够小时终止
   # 向后传递计算协态变量lambda和控制梯度Delta_u
   lambda_hist[:, N-1] = Qn @ xhist[:, N-1]
   for k in range(N-2, -1, -1):
       Delta_u[k] = -(uhist[k] + (1/R) * B.T @ lambda_hist[:, k+1])
       lambda_hist[:, k] = Q @ xhist[:, k] + A.T @ lambda_hist[:, k+1]
   
   # 前向传递与线搜索
   alpha = 1.0
   unew = uhist + alpha * Delta_u
   xnew = rollout(x0, unew)
   
   # Armijo线搜索条件
   while J(xnew, unew) > J(xhist, uhist) - b*alpha*np.dot(Delta_u, Delta_u):
       alpha = 0.5 * alpha
       unew = uhist + alpha * Delta_u
       xnew = rollout(x0, unew)
   
   uhist = unew
   xhist = xnew
   iter_count += 1
   
   if iter_count % 10 == 0:
       print(f"迭代 {iter_count}, 最大梯度: {np.max(np.abs(Delta_u))}, 代价: {J(xhist, uhist)}")
   
   if iter_count > 1000:  # 防止无限循环
       print("达到最大迭代次数")
       break

print(f"总迭代次数: {iter_count}")
print(f"最终代价: {J(xhist, uhist)}")

# 绘制结果
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(thist, xhist[0, :], 'b-', label='位置')
plt.plot(thist, xhist[1, :], 'r-', label='速度')
plt.xlabel('时间')
plt.ylabel('状态')
plt.legend()
plt.title('状态轨迹')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(thist[:-1], uhist, 'g-', label='控制输入')
plt.xlabel('时间')
plt.ylabel('控制')
plt.legend()
plt.title('控制轨迹')
plt.grid(True)

plt.tight_layout()
plt.show()