import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib
import matplotlib.patches as patches

plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 仿真参数
dt = 0.1  # 时间步长
T = 50    # 总时间步数
t = np.arange(T) * dt

# 真实轨迹：圆形运动
omega = 0.5  # 角速度
radius = 5   # 半径
true_x = radius * np.cos(omega * t)
true_y = radius * np.sin(omega * t)
true_vx = -radius * omega * np.sin(omega * t)
true_vy = radius * omega * np.cos(omega * t)

# 系统参数
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])  # 状态转移矩阵

H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])  # 观测矩阵

Q = np.eye(4) * 0.1  # 过程噪声协方差
R = np.eye(2) * 1.0  # 测量噪声协方差

# 生成带噪声的测量数据
measurement_noise = np.random.multivariate_normal([0, 0], R, T)
z_x = true_x + measurement_noise[:, 0]
z_y = true_y + measurement_noise[:, 1]

# 方法1：不使用滤波（直接使用测量值）
no_filter_x = z_x
no_filter_y = z_y
# 通过差分估计速度
no_filter_vx = np.zeros(T)
no_filter_vy = np.zeros(T)
no_filter_vx[1:] = np.diff(no_filter_x) / dt
no_filter_vy[1:] = np.diff(no_filter_y) / dt

# 方法2：卡尔曼滤波器
x_kf = np.array([true_x[0], true_y[0], true_vx[0], true_vy[0]])  # 初始状态
P_kf = np.eye(4) * 1.0  # 初始协方差

# 存储结果
kf_states = np.zeros((T, 4))
kf_covariances = np.zeros((T, 4, 4))

# 卡尔曼滤波递推
for k in range(T):
    # 预测步骤
    x_pred = F @ x_kf
    P_pred = F @ P_kf @ F.T + Q
    
    # 更新步骤
    z_k = np.array([z_x[k], z_y[k]])
    y_k = z_k - H @ x_pred  # 残差
    S_k = H @ P_pred @ H.T + R  # 残差协方差
    K_k = P_pred @ H.T @ np.linalg.inv(S_k)  # 卡尔曼增益
    
    x_kf = x_pred + K_k @ y_k
    P_kf = (np.eye(4) - K_k @ H) @ P_pred
    
    # 存储结果
    kf_states[k] = x_kf
    kf_covariances[k] = P_kf

# 创建可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('卡尔曼滤波 vs 无滤波方法比较', fontsize=16, fontweight='bold')

# 1. 轨迹比较
ax1 = axes[0, 0]
ax1.plot(true_x, true_y, 'g-', linewidth=3, label='真实轨迹', alpha=0.8)
ax1.plot(no_filter_x, no_filter_y, 'r--', linewidth=2, label='无滤波 (直接测量)', alpha=0.7)
ax1.plot(kf_states[:, 0], kf_states[:, 1], 'b-', linewidth=2, label='卡尔曼滤波')
ax1.scatter(z_x[::3], z_y[::3], c='red', s=20, alpha=0.4, label='测量点', zorder=1)
ax1.set_xlabel('X 位置 (m)')
ax1.set_ylabel('Y 位置 (m)')
ax1.set_title('2D 轨迹比较')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# 2. 位置误差比较
ax2 = axes[0, 1]
# 无滤波误差
no_filter_error = np.sqrt((no_filter_x - true_x)**2 + (no_filter_y - true_y)**2)
# 卡尔曼滤波误差
kf_error = np.sqrt((kf_states[:, 0] - true_x)**2 + (kf_states[:, 1] - true_y)**2)

ax2.plot(t, no_filter_error, 'r--', linewidth=2, label='无滤波误差', alpha=0.7)
ax2.plot(t, kf_error, 'b-', linewidth=2, label='卡尔曼滤波误差')
ax2.set_xlabel('时间 (s)')
ax2.set_ylabel('位置误差 (m)')
ax2.set_title('位置估计误差比较')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 速度比较
ax3 = axes[1, 0]
true_speed = np.sqrt(true_vx**2 + true_vy**2)
no_filter_speed = np.sqrt(no_filter_vx**2 + no_filter_vy**2)
kf_speed = np.sqrt(kf_states[:, 2]**2 + kf_states[:, 3]**2)

ax3.plot(t, true_speed, 'g-', linewidth=3, label='真实速度', alpha=0.8)
ax3.plot(t, no_filter_speed, 'r--', linewidth=2, label='无滤波速度估计', alpha=0.7)
ax3.plot(t, kf_speed, 'b-', linewidth=2, label='卡尔曼滤波速度估计')
ax3.set_xlabel('时间 (s)')
ax3.set_ylabel('速度 (m/s)')
ax3.set_title('速度估计比较')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. X和Y位置时间序列比较
ax4 = axes[1, 1]
ax4.plot(t, true_x, 'g-', linewidth=2, label='真实X位置', alpha=0.8)
ax4.plot(t, no_filter_x, 'r--', linewidth=1.5, label='无滤波X位置', alpha=0.7)
ax4.plot(t, kf_states[:, 0], 'b-', linewidth=2, label='卡尔曼滤波X位置')
ax4.set_xlabel('时间 (s)')
ax4.set_ylabel('X 位置 (m)')
ax4.set_title('X位置时间序列比较')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 打印性能统计
print("=" * 50)
print("性能比较统计:")
print("=" * 50)
print(f"平均无滤波位置误差:     {np.mean(no_filter_error):.3f} m")
print(f"平均卡尔曼滤波位置误差: {np.mean(kf_error):.3f} m")
print(f"位置误差改善比例:       {(1 - np.mean(kf_error)/np.mean(no_filter_error))*100:.1f}%")
print()
print(f"无滤波速度估计RMSE:     {np.sqrt(np.mean((no_filter_speed - true_speed)**2)):.3f} m/s")
print(f"卡尔曼滤波速度RMSE:     {np.sqrt(np.mean((kf_speed - true_speed)**2)):.3f} m/s")
print(f"速度估计改善比例:       {(1 - np.sqrt(np.mean((kf_speed - true_speed)**2))/np.sqrt(np.mean((no_filter_speed - true_speed)**2)))*100:.1f}%")
