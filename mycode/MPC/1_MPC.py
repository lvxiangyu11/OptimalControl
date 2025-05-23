import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.linalg import solve_discrete_are
from control import dlqr
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from meshcat import transformations as tf

# ---------------- Matplotlib 中文设置 ----------------
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False   # 正确显示负号

# ---------------- 系统参数 ----------------
g_const = 9.81
m = 1.0
l = 0.3
J = 0.2 * m * l**2

umin = np.array([0.2 * m * g_const, 0.2 * m * g_const])
umax = np.array([0.6 * m * g_const, 0.6 * m * g_const])

h = 0.05
Tfinal = 10.0
Nt = int(Tfinal / h) + 1
time = np.linspace(0, Tfinal, Nt)

# ---------------- 动力学 ----------------
def quad_dynamics(x, u):
    θ = x[2]
    x_dot = np.array([
        x[3],
        x[4],
        x[5],
        (1/m)*(u[0] + u[1])*np.sin(θ),
        (1/m)*(u[0] + u[1])*np.cos(θ) - g_const,
        (1/J)*(l/2)*(u[1] - u[0])
    ])
    return x_dot

def rk4_step(x, u, dt):
    f1 = quad_dynamics(x, u)
    f2 = quad_dynamics(x + 0.5 * dt * f1, u)
    f3 = quad_dynamics(x + 0.5 * dt * f2, u)
    f4 = quad_dynamics(x + dt * f3, u)
    return x + (dt / 6.0) * (f1 + 2*f2 + 2*f3 + f4)

# ---------------- 线性化与 LQR ----------------
x_hover = np.zeros(6)
u_hover = np.array([0.5 * m * g_const, 0.5 * m * g_const])

def linearize_dynamics(f, x0, u0):
    δ = 1e-5
    A = np.zeros((6, 6))
    B = np.zeros((6, 2))
    for i in range(6):
        dx = np.zeros(6)
        dx[i] = δ
        A[:, i] = (rk4_step(x0 + dx, u0, h) - rk4_step(x0 - dx, u0, h)) / (2 * δ)
    for i in range(2):
        du = np.zeros(2)
        du[i] = δ
        B[:, i] = (rk4_step(x0, u0 + du, h) - rk4_step(x0, u0 - du, h)) / (2 * δ)
    return A, B

A, B = linearize_dynamics(quad_dynamics, x_hover, u_hover)
Q = np.eye(6)
R = 0.01 * np.eye(2)
P = solve_discrete_are(A, B, Q, R)
K, _, _ = dlqr(A, B, Q, R)

def lqr_controller(x, x_ref):
    return np.clip(u_hover - K @ (x - x_ref), umin, umax)

# ---------------- 仿真 ----------------
def simulate_closed_loop(x0, controller, x_ref):
    x_hist = np.zeros((6, Nt))
    u_hist = np.zeros((2, Nt-1))
    x = x0.copy()
    for k in range(Nt-1):
        u = controller(x, x_ref)
        x_hist[:, k] = x
        u_hist[:, k] = u
        x = rk4_step(x, u, h)
    x_hist[:, -1] = x
    return x_hist, u_hist

x0 = np.array([10.0, 2.0, 0.0, 0, 0, 0])
x_ref = np.array([0.0, 1.0, 0.0, 0, 0, 0])
x_hist, u_hist = simulate_closed_loop(x0, lqr_controller, x_ref)

# ---------------- 绘图：所有状态和控制量 ----------------
plt.figure(figsize=(10, 6))
plt.plot(time, x_hist[0], label="x位置")
plt.plot(time, x_hist[1], label="y高度")
plt.plot(time, x_hist[2], label="俯仰角θ")
plt.plot(time, x_hist[3], label="x速度")
plt.plot(time, x_hist[4], label="y速度")
plt.plot(time, x_hist[5], label="角速度")
plt.xlabel("时间 [秒]")
plt.ylabel("状态量")
plt.title("四旋翼状态随时间变化")
plt.legend()
plt.grid()

plt.figure(figsize=(8, 4))
plt.plot(time[:-1], u_hist[0], label="左推力 u1")
plt.plot(time[:-1], u_hist[1], label="右推力 u2")
plt.xlabel("时间 [秒]")
plt.ylabel("推力 [N]")
plt.title("控制输入随时间变化")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ---------------- MeshCat 可视化 ----------------
vis = meshcat.Visualizer().open()

# 四旋翼主体
vis["quad"].set_object(g.Box([0.3, 0.05, 0.02]), g.MeshLambertMaterial(color=0x1111ff))

# 两个螺旋桨
vis["quad"]["prop_left"].set_object(g.Sphere(0.02), g.MeshLambertMaterial(color=0xff2222))
vis["quad"]["prop_right"].set_object(g.Sphere(0.02), g.MeshLambertMaterial(color=0xff2222))
vis["quad"]["prop_left"].set_transform(tf.translation_matrix([-0.15, 0, 0]))
vis["quad"]["prop_right"].set_transform(tf.translation_matrix([0.15, 0, 0]))

# 坐标轴：红(x)、绿(y)、蓝(z)
vis["axes"]["x"].set_object(g.Cylinder(0.01, 1.0), g.MeshLambertMaterial(color=0xff0000))
vis["axes"]["x"].set_transform(tf.rotation_matrix(np.pi/2, [0, 0, 1]) @ tf.translation_matrix([0.5, 0, 0]))

vis["axes"]["y"].set_object(g.Cylinder(0.01, 1.0), g.MeshLambertMaterial(color=0x00ff00))
vis["axes"]["y"].set_transform(tf.rotation_matrix(np.pi/2, [0, 0, 1]) @ tf.rotation_matrix(np.pi/2, [0, 1, 0]) @ tf.translation_matrix([0, 0.5, 0]))

vis["axes"]["z"].set_object(g.Cylinder(0.01, 1.0), g.MeshLambertMaterial(color=0x0000ff))
vis["axes"]["z"].set_transform(tf.translation_matrix([0, 0, 0.5]))

import time as pytime

for k in range(Nt):
    y = x_hist[:, k]
    pos = [0, y[0], y[1]]  # 平面中 (x, y)
    rot = tf.rotation_matrix(-y[2], [0, 0, 1])  # 绕z轴旋转
    transform = tf.concatenate_matrices(tf.translation_matrix(pos), rot)
    vis["quad"].set_transform(transform)
    pytime.sleep(0.02)
