import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
import matplotlib.font_manager as fm
plt.rcParams["font.sans-serif"]=["SimHei"] 
plt.rcParams["axes.unicode_minus"]=False

# 设置参数
g = 9.81  # 重力加速度
l = 1.0   # 摆长
theta = 0  # 分析平衡点θ=0处的稳定性

# 定义连续系统的A矩阵
def A_continuous(theta):
    return np.array([
        [0, 1],
        [-g/l * np.cos(theta), 0]
    ])

# 定义离散系统的A矩阵（使用欧拉方法）
def A_discrete(h, theta):
    A_cont = A_continuous(theta)
    I = np.eye(2)
    return I + h * A_cont

# 计算不同时间步长下的特征值
h_values = np.linspace(0.01, 0.5, 100)
eigenvalues = []

for h in h_values:
    A_d = A_discrete(h, theta)
    eigs = eigvals(A_d)
    eigenvalues.append(eigs)

# 转换为numpy数组便于处理
eigenvalues = np.array(eigenvalues)

# 计算特征值的模
magnitudes = np.abs(eigenvalues)

# 创建图形
plt.figure(figsize=(14, 8))

# 绘制特征值模随h的变化
plt.subplot(2, 2, 1)
plt.plot(h_values, magnitudes[:, 0], 'b-', label='|λ1|')
plt.plot(h_values, magnitudes[:, 1], 'r--', label='|λ2|')
plt.axhline(y=1, color='k', linestyle='-', alpha=0.3)
plt.grid(True)
plt.xlabel('时间步长 h')
plt.ylabel('特征值模 |λ|')
plt.title('特征值模 vs 时间步长')
plt.legend()

# 绘制实部随h的变化
plt.subplot(2, 2, 2)
plt.plot(h_values, np.real(eigenvalues[:, 0]), 'b-', label='Re(λ1)')
plt.plot(h_values, np.real(eigenvalues[:, 1]), 'r--', label='Re(λ2)')
plt.grid(True)
plt.xlabel('时间步长 h')
plt.ylabel('特征值实部 Re(λ)')
plt.title('特征值实部 vs 时间步长')
plt.legend()

# 绘制虚部随h的变化
plt.subplot(2, 2, 3)
plt.plot(h_values, np.imag(eigenvalues[:, 0]), 'b-', label='Im(λ1)')
plt.plot(h_values, np.imag(eigenvalues[:, 1]), 'r--', label='Im(λ2)')
plt.grid(True)
plt.xlabel('时间步长 h')
plt.ylabel('特征值虚部 Im(λ)')
plt.title('特征值虚部 vs 时间步长')
plt.legend()

# 绘制特征值在复平面上的分布
plt.subplot(2, 2, 4)
for i, h in enumerate(h_values):
    if i % 5 == 0:  # 每5个点绘制一次，避免过于密集
        plt.plot(np.real(eigenvalues[i, 0]), np.imag(eigenvalues[i, 0]), 'bo', alpha=0.5)
        plt.plot(np.real(eigenvalues[i, 1]), np.imag(eigenvalues[i, 1]), 'ro', alpha=0.5)

# 绘制单位圆
theta_circle = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(theta_circle), np.sin(theta_circle), 'k-', alpha=0.3)
plt.grid(True)
plt.axis('equal')
plt.xlabel('实部 Re(λ)')
plt.ylabel('虚部 Im(λ)')
plt.title('特征值在复平面上的分布')

# 特别标记h=0.1时的特征值
h_01_index = np.argmin(np.abs(h_values - 0.1))
h_01 = h_values[h_01_index]
A_d_01 = A_discrete(h_01, theta)
eigs_01 = eigvals(A_d_01)
magnitude_01 = np.abs(eigs_01)

plt.subplot(2, 2, 4)
plt.plot(np.real(eigs_01[0]), np.imag(eigs_01[0]), 'bs', markersize=8, label=f'h=0.1, |λ1|={magnitude_01[0]:.3f}')
plt.plot(np.real(eigs_01[1]), np.imag(eigs_01[1]), 'rs', markersize=8, label=f'h=0.1, |λ2|={magnitude_01[1]:.3f}')
plt.legend()

# 添加总标题，包含稳定性分析结果
plt.suptitle(f'简单单摆系统欧拉法离散化稳定性分析\n'
             f'h=0.1时: λ = {eigs_01[0]:.3f}, {eigs_01[1]:.3f}; '
             f'|λ| = {magnitude_01[0]:.3f}, {magnitude_01[1]:.3f} > 1 → 不稳定', 
             fontsize=14)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

# 验证h=0.1时的特征值
h_test = 0.1
A_d_test = A_discrete(h_test, theta)
eigs_test = eigvals(A_d_test)
print(f"h=0.1时的特征值:")
print(f"λ1 = {eigs_test[0]:.6f}, |λ1| = {abs(eigs_test[0]):.6f}")
print(f"λ2 = {eigs_test[1]:.6f}, |λ2| = {abs(eigs_test[1]):.6f}")