"""LQR 目标跟踪示例 (Franka Panda in PyBullet)

说明 (中文):
	本脚本演示如何将任意目标状态跟踪问题转换为标准 LQR 问题:
		u = u_goal - K (x - x_goal)  (等价写成 u = -K(x-x_goal) + u_goal)

	我们使用 Franka Panda 机械臂 (7 自由度), 状态定义为:
		x = [q; qdot] ∈ R^{14}
	控制输入为关节力矩 u ∈ R^{7}

	选择离散时间步长 dt, 线性化的近似动态 (在平衡点附近):
		q_{k+1}   = q_k + qdot_k * dt
		qdot_{k+1}= qdot_k + M(q*)^{-1} ( u_k - u_goal - ( ∂g/∂q |_{q*} (q_k - q*) ) ) * dt  (忽略科氏/阻尼项小扰动)

	进一步近似 (忽略 ∂g/∂q 项) 得到简化线性系统:
		x_{k+1} = A x_k + B u_k
	  其中:
		A = [[I, dt*I],[0, I]]  (维度 14x14)
		B = [[0],[ M^{-1} dt ]] (维度 14x7)

	更精确的做法是数值微分重力项 g(q) 得到 ∂g/∂q, 从而改进 A 矩阵; 这里提供可切换选项 use_gravity_jacobian。

	平衡点 (x_goal, u_goal):
		令 qdot_goal = 0, u_goal = g(q_goal)  (静态平衡: M * 0 + g - u_goal = 0)
	这样在偏差变量 \tilde{x} = x - x_goal, \tilde{u} = u - u_goal 下得到标准 LQR。

依赖:
	pip install pybullet numpy

运行:
	python 4_LQR_Target_Tracking_pybullet.py --ik --target 0.5 0.0 0.4

参数:
	--ik 使用笛卡尔末端位置 (通过逆解 IK) 指定目标; 否则用 --qgoal 直接提供 7 个关节角度.
	--track_steps 在模拟过程中是否定期重新线性化 (时变 LQR). 默认只在初始点线性化一次。

注意:
	这是教学/演示代码, 并未包含关节力矩限制、碰撞避免、摩擦/阻尼精确建模等; 目标姿态过远可能导致线性化近似误差大、震荡或不收敛。

Author: 自动生成 (GitHub Copilot)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pybullet as p
import pybullet_data


# --------------------------- 工具函数 --------------------------- #

def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, max_iter: int = 500, eps: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
	"""离散时间无限时域 LQR 求解器 (通过迭代求解 DARE)。

	返回:
		K: 反馈增益
		P: 代价函数的黎卡提矩阵
	"""
	P = Q.copy()
	for _ in range(max_iter):
		P_next = A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
		if np.max(np.abs(P_next - P)) < eps:
			P = P_next
			break
		P = P_next
	K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
	return K, P


def numerical_jacobian(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
	"""数值雅可比 (列为每个变量的偏导)."""
	x = x.astype(float)
	fx = f(x)
	m = fx.shape[0]
	n = x.shape[0]
	J = np.zeros((m, n))
	for i in range(n):
		xpert = x.copy()
		xpert[i] += eps
		J[:, i] = (f(xpert) - fx) / eps
	return J


def get_mass_matrix(robot_id: int, joint_indices) -> np.ndarray:
	"""
	获取关节质量矩阵 M(q)。优先用 pybullet 的 calculateMassMatrix（更安全），否则用 inverseDynamics trick。
	"""
	n = len(joint_indices)
	q = [p.getJointState(robot_id, j)[0] for j in joint_indices]
	# 优先尝试 pybullet 的 calculateMassMatrix（部分 pybullet 版本支持）
	try:
		M = p.calculateMassMatrix(robot_id, q)
		M = np.array(M)
		if M.shape == (n, n):
			return M
	except Exception:
		pass
	# 回退方案：inverseDynamics trick，但要补齐所有自由度
	# 获取全部可控关节数
	num_joints = p.getNumJoints(robot_id)
	all_indices = [j for j in range(num_joints) if p.getJointInfo(robot_id, j)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)]
	q_full = [p.getJointState(robot_id, j)[0] for j in all_indices]
	qd_full = [0.0] * len(all_indices)
	g_full = p.calculateInverseDynamics(robot_id, q_full, qd_full, [0.0] * len(all_indices))
	# 只取前 n 维
	M = np.zeros((n, n))
	eye = np.eye(n)
	for i in range(n):
		ddq_full = [0.0] * len(all_indices)
		ddq_full[i] = 1.0
		tau_full = p.calculateInverseDynamics(robot_id, q_full, qd_full, ddq_full)
		ddq_full = [1.0] * len(all_indices)
		# tau = M * ddq + g  => M_col_i = (tau - g)/ ddq
		M[:, i] = (np.array(tau_full[:n]) - np.array(g_full[:n])) / np.array(ddq_full[:n])
	return M


def get_gravity_torque(robot_id: int, joint_indices) -> np.ndarray:
	"""
	获取重力力矩。为兼容 pybullet，需传递所有可控关节的 q、qd、ddq，最后只返回前 n (arm) 维。
	"""
	n = len(joint_indices)
	num_joints = p.getNumJoints(robot_id)
	all_indices = [j for j in range(num_joints) if p.getJointInfo(robot_id, j)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)]
	q_full = [p.getJointState(robot_id, j)[0] for j in all_indices]
	qd_full = [0.0] * len(all_indices)
	ddq_full = [0.0] * len(all_indices)
	tau_full = p.calculateInverseDynamics(robot_id, q_full, qd_full, ddq_full)
	return np.array(tau_full[:n])


def gravity_jacobian(robot_id: int, joint_indices, q_star: np.ndarray, eps: float = 1e-4) -> np.ndarray:
	'''数值计算重力项 g(q) 关于关节角 q 的雅可比矩阵。'''
	# 切换到给定 q_star, 暂存当前状态
	saved = [p.getJointState(robot_id, j)[0] for j in joint_indices]
	for j, val in zip(joint_indices, q_star):
		p.resetJointState(robot_id, j, val, targetVelocity=0.0)

	def g_func(qvec):
		for j, val in zip(joint_indices, qvec):
			p.resetJointState(robot_id, j, val, targetVelocity=0.0)
		return get_gravity_torque(robot_id, joint_indices)

	Jg = numerical_jacobian(g_func, q_star, eps=eps)
	# 还原
	for j, val in zip(joint_indices, saved):
		p.resetJointState(robot_id, j, val, targetVelocity=0.0)
	return Jg


# --------------------------- 数据类 --------------------------- #

@dataclass
class LQRConfig:
	dt: float = 0.002
	Q_pos: float = 200.0  # 位置误差权重 (对角)
	Q_vel: float = 5.0    # 速度误差权重
	R: float = 0.01       # 力矩权重
	use_gravity_jacobian: bool = True
	relinearize_every: int = 400  # 周期性重新线性化 (步数); <=0 表示不重新线性化
	torque_limit: float = 60.0    # 力矩限幅 (绝对值)
	sat_relinearize_window: int = 400  # 在窗口内饱和且误差未改善则强制重新线性化
	sat_improve_eps: float = 1e-3
	adaptive_gravity: bool = True  # 每步使用 g(q) 而不是固定 g(q_goal)
	auto_scale_q: bool = True      # 长期饱和误差不减时自动降低 Q_pos
	auto_scale_factor: float = 0.5
	auto_scale_min_qpos: float = 5.0


# --------------------------- 主逻辑 --------------------------- #

def build_linear_model(robot_id: int, joint_indices, q_goal: np.ndarray, cfg: LQRConfig):
	n = len(joint_indices)
	dt = cfg.dt
	# 记录当前仿真中的关节状态, 线性化时临时切到 q_goal, 结束后恢复，避免“初始化就已经在目标点”现象
	orig_q = [p.getJointState(robot_id, j)[0] for j in joint_indices]
	orig_qd = [p.getJointState(robot_id, j)[1] for j in joint_indices]
	for j, val in zip(joint_indices, q_goal):
		p.resetJointState(robot_id, j, val, targetVelocity=0.0)

	M = get_mass_matrix(robot_id, joint_indices)
	g_vec = get_gravity_torque(robot_id, joint_indices)
	if cfg.use_gravity_jacobian:
		Jg = gravity_jacobian(robot_id, joint_indices, q_goal.copy())  # shape (n,n)
	else:
		Jg = np.zeros((n, n))

	Minv = np.linalg.inv(M)

	# 状态: x = [q; qd]
	A = np.zeros((2 * n, 2 * n))
	A[:n, :n] = np.eye(n)
	A[:n, n:] = dt * np.eye(n)
	# qdot_{k+1} 部分: qdot + dt * Minv ( -Jg (q - q*) - ... + u - u*) => 对 q 偏导 -dt*Minv*Jg, 对 qdot 偏导 I
	A[n:, :n] = -dt * Minv @ Jg
	A[n:, n:] = np.eye(n)

	B = np.zeros((2 * n, n))
	B[n:, :] = dt * Minv  # 仅加速度受控制

	# 代价矩阵
	Q = np.zeros((2 * n, 2 * n))
	Q[:n, :n] = cfg.Q_pos * np.eye(n)
	Q[n:, n:] = cfg.Q_vel * np.eye(n)
	R = cfg.R * np.eye(n)

	K, P = dlqr(A, B, Q, R)

	# 恢复原始关节位置与速度，确保主循环从用户设置的初始状态开始运动
	for j, qv, qdv in zip(joint_indices, orig_q, orig_qd):
		p.resetJointState(robot_id, j, qv, targetVelocity=qdv)

	print("A =", A, "B =", B, "K =", K, "P =", P, "g_vec =", g_vec)
	return A, B, K, P, g_vec


def load_panda(gui: bool = True):
	cid = p.connect(p.GUI if gui else p.DIRECT)
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=50, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.6])
	p.setGravity(0, 0, -9.81)
	plane = p.loadURDF("plane.urdf")
	flags = p.URDF_USE_SELF_COLLISION
	robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, flags=flags)
	# 获取可控关节 (排除手指等)
	joint_indices = []
	for j in range(p.getNumJoints(robot)):
		info = p.getJointInfo(robot, j)
		joint_type = info[2]
		name = info[1].decode()
		if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC) and name.startswith("panda_joint"):
			joint_indices.append(j)
	joint_indices = sorted(joint_indices)[:7]
	return robot, joint_indices


def current_state(robot_id: int, joint_indices):
	q = []
	qd = []
	for j in joint_indices:
		st = p.getJointState(robot_id, j)
		q.append(st[0])
		qd.append(st[1])
	return np.array(q), np.array(qd)


def apply_torque(robot_id: int, joint_indices, tau: np.ndarray):
	p.setJointMotorControlArray(robot_id, joint_indices, p.TORQUE_CONTROL, forces=tau.tolist())


def inverse_kinematics_to_q(robot_id: int, target_pos, target_orn=None):
	if target_orn is None:
		# 末端默认保持原始姿态 (使用当前关节角计算)
		pass
	end_effector_index = 11  # Panda gripper link index (often 11)
	if target_orn is not None:
		q_sol = p.calculateInverseKinematics(robot_id, end_effector_index, targetPosition=target_pos, targetOrientation=target_orn)
	else:
		q_sol = p.calculateInverseKinematics(robot_id, end_effector_index, targetPosition=target_pos)
	return np.array(q_sol)[:7]


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--nogui', action='store_true', help='不使用 GUI')
	parser.add_argument('--dt', type=float, default=0.002)
	parser.add_argument('--time', type=float, default=5.0, help='模拟时间 (s)')
	parser.add_argument('--ik', action='store_true', help='使用末端笛卡尔坐标作为目标 (逆解)')
	parser.add_argument('--target', type=float, nargs=3, default=[0.5, 0.0, 0.5], help='末端目标位置 (x y z)')
	parser.add_argument('--qgoal', type=float, nargs=7, help='直接指定 7 个关节角目标 (rad)')
	parser.add_argument('--relinearize', type=int, default=0, help='>0 则每隔 N 步重新线性化')
	parser.add_argument('--use_gravity_jacobian', action='store_true')
	parser.add_argument('--qpos_weight', type=float, default=200.0, help='Q_pos 权重')
	parser.add_argument('--qvel_weight', type=float, default=5.0, help='Q_vel 权重')
	parser.add_argument('--r_weight', type=float, default=0.01, help='R 权重(力矩)')
	parser.add_argument('--torque_limit', type=float, default=60.0, help='力矩限幅 (Nm)')
	parser.add_argument('--sat_relinearize_window', type=int, default=400, help='窗口大小: 若饱和且误差未降则强制重新线性化')
	parser.add_argument('--sat_improve_eps', type=float, default=1e-3, help='误差改善阈值')
	parser.add_argument('--no_adaptive_gravity', action='store_true', help='关闭逐步重力补偿 (默认开启)')
	parser.add_argument('--no_auto_scale_q', action='store_true', help='关闭 Q_pos 自动缩放')
	args = parser.parse_args()

	cfg = LQRConfig(dt=args.dt,
		use_gravity_jacobian=args.use_gravity_jacobian,
		relinearize_every=args.relinearize,
		Q_pos=args.qpos_weight,
		Q_vel=args.qvel_weight,
		R=args.r_weight,
		torque_limit=args.torque_limit,
		sat_relinearize_window=args.sat_relinearize_window,
		sat_improve_eps=args.sat_improve_eps,
		adaptive_gravity=not args.no_adaptive_gravity,
		auto_scale_q=not args.no_auto_scale_q)

	robot_id, joint_indices = load_panda(gui=not args.nogui)
	# 关闭默认位置/速度控制器确保纯力矩控制
	for j in joint_indices:
		p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0)

	# 初始关节角: 轻微弯曲
	q_init = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.8, 0.8])
	for j, val in zip(joint_indices, q_init):
		p.resetJointState(robot_id, j, val, targetVelocity=0.0)

	if args.ik:
		q_goal = inverse_kinematics_to_q(robot_id, args.target)
	elif args.qgoal is not None:
		q_goal = np.array(args.qgoal)
	else:
		q_goal = np.array([0.5, -1.0, 0.8, -2.5, 0.5, 2.2, 1.2])

	# 平衡速度 = 0
	qd_goal = np.zeros_like(q_goal)

	# 计算线性模型 & LQR 增益
	A, B, K, P, g_goal = build_linear_model(robot_id, joint_indices, q_goal, cfg)
	u_goal = g_goal  # 平衡力矩 (目标点重力补偿)

	print('q_goal:', q_goal)
	print('u_goal (gravity compensation):', u_goal)
	print('LQR gain K shape:', K.shape)

	steps = int(args.time / cfg.dt)
	trace_err = []
	sat_count = 0
	best_err = np.inf
	last_relin = 0
	for k in range(steps):
		q, qd = current_state(robot_id, joint_indices)
		x = np.hstack([q, qd])
		x_goal = np.hstack([q_goal, qd_goal])
		err = x - x_goal
		# 自适应重力补偿：用当前 g(q) 替换固定 g(q_goal)
		if cfg.adaptive_gravity:
			g_curr = get_gravity_torque(robot_id, joint_indices)
			feedforward = g_curr
		else:
			feedforward = u_goal
		# LQR 控制律: u = g(q) - K (x - x_goal)
		u = feedforward - K @ err

		# 力矩限幅
		u_clipped = np.clip(u, -cfg.torque_limit, cfg.torque_limit)
		was_saturated = np.any(np.abs(u) > cfg.torque_limit * 0.999)
		u = u_clipped

		apply_torque(robot_id, joint_indices, u)
		if k % 50 == 0:
			print(f'Step {k}: u={np.round(u,3)}')
		p.stepSimulation()

		pos_err_norm = np.linalg.norm(q - q_goal)
		vel_err_norm = np.linalg.norm(qd - qd_goal)
		if k % 50 == 0:
			print(f'step {k}: |dq|={pos_err_norm:.4f}, |dqd|={vel_err_norm:.4f}')

		trace_err.append(pos_err_norm)
		if pos_err_norm < best_err - cfg.sat_improve_eps:
			best_err = pos_err_norm
			# 错误有改进，重置计数
			sat_count = 0
		elif was_saturated:
			sat_count += 1
		else:
			# 未饱和但也没改进，适当累积较慢
			sat_count = max(0, sat_count - 1)

		# 如果长时间饱和且误差未改善，强制重新线性化
		if (cfg.relinearize_every <= 0) and cfg.sat_relinearize_window > 0 and sat_count >= cfg.sat_relinearize_window:
			print(f'[sat-relinearize @ step {k}] 长时间饱和且误差停滞 -> 重新线性化')
			A, B, K, P, g_goal = build_linear_model(robot_id, joint_indices, q_goal, cfg)
			u_goal = g_goal
			sat_count = 0
			last_relin = k
			# 自动缩放 Q_pos 降低增益，帮助脱离饱和
			if cfg.auto_scale_q and cfg.Q_pos > cfg.auto_scale_min_qpos:
				cfg.Q_pos = max(cfg.auto_scale_min_qpos, cfg.Q_pos * cfg.auto_scale_factor)
				print(f'[auto-scale] 新 Q_pos={cfg.Q_pos:.2f}, 重新求解 LQR')
				A, B, K, P, g_goal = build_linear_model(robot_id, joint_indices, q_goal, cfg)
				u_goal = g_goal

		# 可选重新线性化 (时变 LQR)
		if cfg.relinearize_every > 0 and (k - last_relin) >= cfg.relinearize_every:
			A, B, K, P, g_goal = build_linear_model(robot_id, joint_indices, q_goal, cfg)
			u_goal = g_goal
			last_relin = k
			print(f'[relinearize @ step {k}] recomputed K')

		# 收敛判据：位置和速度误差都很小则提前停止
		if pos_err_norm < 1e-3 and vel_err_norm < 1e-3:
			print(f'Converged at step {k}: |dq|={pos_err_norm:.6f}, |dqd|={vel_err_norm:.6f}')
			break

		if not args.nogui:
			time.sleep(cfg.dt)

	print('Final position error norm:', trace_err[-1])
	if not args.nogui:
		print('保持窗口 3 秒...')
		time.sleep(3)
	p.disconnect()


if __name__ == '__main__':
	main()

