# ============================================================================
# Acrobot 最优控制问题 - 使用差分动态规划(DDP)算法求解
# 
# 问题描述：
# Acrobot是一个欠驱动的双摆系统，只有第二个关节有驱动力矩，第一个关节无驱动。
# 目标是将系统从悬垂位置控制到倒立位置。
#
# 算法：差分动态规划(Differential Dynamic Programming, DDP)
# - DDP是一种基于二阶导数信息的轨迹优化方法
# - 通过迭代优化来找到最优控制序列
# ============================================================================

# 环境设置 - 激活当前目录的包环境并安装依赖
import Pkg; Pkg.activate(@__DIR__); 
Pkg.instantiate()

# 导入所需的包
using LinearAlgebra  # 线性代数运算
using PyPlot        # 绘图
using ForwardDiff   # 自动微分，用于计算雅可比矩阵
using MeshCat       # 3D可视化

# ============================================================================
# Acrobot 动力学模型
# ============================================================================
function acrobot_dynamics(x, u)
    """
    Acrobot系统的连续时间动力学方程
    
    参数：
    - x: 状态向量 [θ1, θ2, θ1dot, θ2dot]
      θ1: 第一个关节角度 (无驱动)
      θ2: 第二个关节角度 (有驱动) 
      θ1dot, θ2dot: 对应的角速度
    - u: 控制输入 [τ2] (只有第二个关节的力矩)
    
    返回：状态导数 [θ1dot, θ2dot, θ1ddot, θ2ddot]
    """
    
    # 系统物理参数
    g = 9.81        # 重力加速度 (m/s²)
    m1, m2 = [1., 1.]  # 连杆质量 (kg)
    l1, l2 = [1., 1.]  # 连杆长度 (m)  
    J1, J2 = [1., 1.]  # 连杆转动惯量 (kg⋅m²)
    
    # 提取状态变量
    θ1,    θ2    = x[1], x[2]      # 关节角度
    θ1dot, θ2dot = x[3], x[4]      # 关节角速度
    
    # 三角函数预计算（提高计算效率）
    s1, c1 = sincos(θ1)           # sin(θ1), cos(θ1)
    s2, c2 = sincos(θ2)           # sin(θ2), cos(θ2)  
    c12 = cos(θ1 + θ2)            # cos(θ1 + θ2)

    # 质量矩阵 M (惯性矩阵)
    # 描述系统的惯性特性，来源于拉格朗日方程
    m11 = m1*l1^2 + J1 + m2*(l1^2 + l2^2 + 2*l1*l2*c2) + J2  # (1,1)元素
    m12 = m2*(l2^2 + l1*l2*c2 + J2)                           # (1,2)元素
    m22 = l2^2*m2 + J2                                        # (2,2)元素
    M = [m11 m12; m12 m22]  # 2×2对称正定矩阵

    # 科里奥利力和离心力项 B
    # 由于关节速度耦合产生的非线性力
    tmp = l1*l2*m2*s2
    b1 = -(2 * θ1dot * θ2dot + θ2dot^2)*tmp  # 第一个关节的科里奥利力
    b2 = tmp * θ1dot^2                        # 第二个关节的科里奥利力
    B = [b1, b2]

    # 摩擦力项 C (简化的粘性摩擦模型)
    c = 1.0  # 摩擦系数
    C = [c*θ1dot, c*θ2dot]  # 与角速度成正比的摩擦力

    # 重力项 G
    # 由重力产生的力矩
    g1 = ((m1 + m2)*l1*c1 + m2*l2*c12) * g  # 第一个关节的重力力矩
    g2 = m2*l2*c12*g                         # 第二个关节的重力力矩
    G = [g1, g2]

    # 运动方程求解: M*θddot = τ - B - G - C
    # τ = [0, u[1]] 表示只有第二个关节有外加力矩
    τ = [0, u[1]]
    θddot = M\(τ - B - G - C)  # 解线性方程组得到角加速度
    
    # 返回状态导数 [θ1dot, θ2dot, θ1ddot, θ2ddot]
    return [θ1dot, θ2dot, θddot[1], θddot[2]]
end

# ============================================================================
# 数值积分 - 四阶龙格-库塔法 (RK4)
# ============================================================================
function dynamics_rk4(x, u)
    """
    使用四阶龙格-库塔方法进行数值积分
    将连续时间动力学离散化为离散时间系统
    
    RK4是一种高精度的数值积分方法，误差阶为O(h^5)
    对控制输入u采用零阶保持(Zero-Order Hold)
    """
    # RK4的四个斜率计算
    f1 = acrobot_dynamics(x, u)                    # k1 = f(x, u)
    f2 = acrobot_dynamics(x + 0.5*h*f1, u)        # k2 = f(x + h/2*k1, u)
    f3 = acrobot_dynamics(x + 0.5*h*f2, u)        # k3 = f(x + h/2*k2, u)  
    f4 = acrobot_dynamics(x + h*f3, u)            # k4 = f(x + h*k3, u)
    
    # RK4组合公式: x_{k+1} = x_k + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
end

# ============================================================================
# 雅可比矩阵计算 - 使用自动微分
# ============================================================================
function dfdx(x, u)
    """
    计算动力学函数关于状态x的雅可比矩阵 A = ∂f/∂x
    这是DDP算法中线性化所需的矩阵
    """
    ForwardDiff.jacobian(dx->dynamics_rk4(dx, u), x)
end

function dfdu(x, u)
    """
    计算动力学函数关于控制u的雅可比矩阵 B = ∂f/∂u  
    这是DDP算法中线性化所需的矩阵
    """
    ForwardDiff.derivative(du->dynamics_rk4(x, du), u)
end

# ============================================================================
# 问题参数设置
# ============================================================================
h = 0.05        # 时间步长 (秒)
Nx = 4          # 状态维数 [θ1, θ2, θ1dot, θ2dot]
Nu = 1          # 控制维数 [τ2]
Tfinal = 10.0   # 总时间 (秒)
Nt = Int(Tfinal/h)+1    # 时间步数
thist = Array(range(0, h*(Nt-1), step=h));  # 时间向量

# ============================================================================
# 代价函数权重矩阵设计
# ============================================================================
# 阶段代价权重矩阵Q - 对不同状态分量给予不同的权重
Q = Diagonal([1.0*ones(2); 0.1*ones(2)]);  # 位置权重大，速度权重小
R = 0.01;                                   # 控制代价权重（能耗考虑）
Qn = Array(100.0*I(Nx));                   # 终端代价权重（终端约束强化）

# ============================================================================
# 代价函数定义
# ============================================================================
function stage_cost(x, u)
    """
    阶段代价函数 l(x,u) = 0.5*(x-x_goal)^T*Q*(x-x_goal) + 0.5*R*u^2
    
    - 状态跟踪项：惩罚偏离目标状态
    - 控制正则化项：惩罚过大的控制输入，节约能耗
    """
    return 0.5*((x-xgoal)'*Q*(x-xgoal)) + 0.5*R*u*u
end

function terminal_cost(x)
    """
    终端代价函数 φ(x_N) = 0.5*(x_N-x_goal)^T*Q_N*(x_N-x_goal)
    
    强化终端时刻对目标状态的约束
    """
    return 0.5*(x-xgoal)'*Qn*(x-xgoal)
end

function cost(xtraj, utraj)
    """
    总代价函数 J = Σ_{k=0}^{N-1} l(x_k,u_k) + φ(x_N)
    """
    J = 0.0
    # 累积阶段代价
    for k = 1:(Nt-1)
        J += stage_cost(xtraj[:,k], utraj[k])
    end
    # 添加终端代价
    J += terminal_cost(xtraj[:,Nt])
    return J
end

# ============================================================================
# 初始条件和目标设置
# ============================================================================
x0 = [-pi/2; 0; 0; 0]      # 初始状态：悬垂位置 (向下)
xgoal = [pi/2; 0; 0; 0]    # 目标状态：倒立位置 (向上)

# 初始轨迹猜测
xtraj = kron(ones(1,Nt), x0)  # 将初始状态复制Nt次作为初始轨迹
utraj = randn(Nt-1);          # 随机初始化控制序列

# ============================================================================
# 初始前向积分
# ============================================================================
# 根据初始控制序列计算对应的状态轨迹
for k = 1:(Nt-1)
    xtraj[:,k+1] .= dynamics_rk4(xtraj[:,k], utraj[k])
end
J = cost(xtraj, utraj)  # 计算初始代价

# ============================================================================
# DDP算法主体实现
# ============================================================================
using Printf

# DDP算法所需的变量初始化
p = zeros(Nx, Nt)           # 协状态向量(拉格朗日乘子)
P = zeros(Nx, Nx, Nt)       # 值函数的二阶导数矩阵
d = ones(Nt-1)              # 前馈控制修正
K = zeros(Nu, Nx, Nt-1)     # 反馈增益矩阵
ΔJ = 0.0                    # 预期代价改善量

# 新轨迹存储
xn = zeros(Nx, Nt)
un = zeros(Nt-1)

# 梯度和Hessian矩阵
gx = zeros(Nx)              # 哈密顿函数关于x的梯度
gu = 0.0                    # 哈密顿函数关于u的梯度
Gxx = zeros(Nx, Nx)         # Q函数关于x的Hessian
Guu = 0.0                   # Q函数关于u的Hessian
Gxu = zeros(Nx)             # Q函数的混合二阶导数
Gux = zeros(Nx)             # Q函数的混合二阶导数

iter = 0
# ============================================================================
# DDP主迭代循环
# ============================================================================
while maximum(abs.(d[:])) > 1e-3  # 收敛判据：前馈修正足够小
    iter += 1
    
    # 重新初始化
    p = zeros(Nx, Nt)
    P = zeros(Nx, Nx, Nt)
    d = ones(Nt-1)
    K = zeros(Nu, Nx, Nt-1)
    ΔJ = 0.0

    # 终端条件设置（来自终端代价的导数）
    p[:,Nt] = Qn*(xtraj[:,Nt]-xgoal)     # ∂φ/∂x
    P[:,:,Nt] = Qn                        # ∂²φ/∂x²
    
    # ========================================================================
    # 后向传播 (Backward Pass)
    # ========================================================================
    # 从终端时刻向前传播，计算最优控制策略
    for k = (Nt-1):-1:1
        # 计算阶段代价的一阶和二阶导数
        q = Q*(xtraj[:,k]-xgoal)         # ∂l/∂x
        r = R*utraj[k]                   # ∂l/∂u
    
        # 计算动力学的雅可比矩阵（线性化）
        A = dfdx(xtraj[:,k], utraj[k])   # ∂f/∂x
        B = dfdu(xtraj[:,k], utraj[k])   # ∂f/∂u
    
        # 计算Q函数的梯度（哈密顿函数的导数）
        gx = q + A'*p[:,k+1]             # ∂Q/∂x
        gu = r + B'*p[:,k+1]             # ∂Q/∂u
    
        # 计算Q函数的Hessian矩阵（二阶导数）
        Gxx = Q + A'*P[:,:,k+1]*A        # ∂²Q/∂x²
        Guu = R + B'*P[:,:,k+1]*B        # ∂²Q/∂u²
        Gxu = A'*P[:,:,k+1]*B            # ∂²Q/∂x∂u
        Gux = B'*P[:,:,k+1]*A            # ∂²Q/∂u∂x
        
        # 计算最优控制策略
        d[k] = Guu\gu                    # 前馈项：δu = -G_uu^(-1) * g_u
        K[:,:,k] = Guu\Gux               # 反馈项：δu = -K*δx
    
        # 更新协状态和值函数Hessian（Riccati递推）
        p[:,k] = gx - K[:,:,k]'*gu + K[:,:,k]'*Guu*d[k] - Gxu*d[k]
        P[:,:,k] = Gxx + K[:,:,k]'*Guu*K[:,:,k] - Gxu*K[:,:,k] - K[:,:,k]'*Gux
    
        # 累积预期代价改善量
        ΔJ += gu'*d[k]
    end

    # ========================================================================
    # 前向积分与线搜索 (Forward Pass with Line Search)
    # ========================================================================
    # 应用计算得到的控制策略，生成新的轨迹
    xn[:,1] = xtraj[:,1]    # 初始状态保持不变
    α = 1.0                 # 线搜索步长初始化
    
    # 前向积分：应用新的控制策略
    for k = 1:(Nt-1)
        # 新控制 = 旧控制 - α*前馈修正 - 反馈增益*状态偏差
        un[k] = utraj[k] - α*d[k] - dot(K[:,:,k], xn[:,k]-xtraj[:,k])
        xn[:,k+1] .= dynamics_rk4(xn[:,k], un[k])
    end
    Jn = cost(xn, un)  # 计算新轨迹的代价
    
    # 线搜索：Armijo条件检验
    # 如果新代价不满足充分下降条件，则减小步长
    while isnan(Jn) || Jn > (J - 1e-2*α*ΔJ)
        α = 0.5*α  # 步长减半
        for k = 1:(Nt-1)
            un[k] = utraj[k] - α*d[k] - dot(K[:,:,k], xn[:,k]-xtraj[:,k])
            xn[:,k+1] = dynamics_rk4(xn[:,k], un[k])
        end
        Jn = cost(xn, un)
    end
    
    # ========================================================================
    # 迭代信息输出
    # ========================================================================
    if rem(iter - 1, 100) == 0
        @printf "iter     J           ΔJ        |d|         α       \n"
        @printf "---------------------------------------------------\n"
    end
    if rem(iter - 1, 10) == 0 
        @printf("%3d   %10.3e  %9.2e  %9.2e  %6.4f  \n",
              iter, J, ΔJ, maximum(abs.(d[:])), α)
    end
    
    # 更新轨迹
    J = Jn
    xtraj .= xn
    utraj .= un
end

# ============================================================================
# 结果可视化
# ============================================================================
# 绘制关节角度轨迹
plot(thist, xtraj[1,:])  # θ1轨迹
plot(thist, xtraj[2,:])  # θ2轨迹

# 绘制控制输入
plot(thist[1:Nt-1], utraj)  # 控制力矩轨迹

# ============================================================================
# 3D动画可视化
# ============================================================================
using Colors

function build_acrobot!(vis, color=colorant"blue", thick=0.05)
    """
    构建Acrobot的3D可视化模型
    
    参数：
    - vis: MeshCat可视化器
    - color: 连杆颜色
    - thick: 连杆粗细
    """
    l1, l2 = [1., 1.]  # 连杆长度
    
    # 创建几何体
    hinge = MeshCat.Cylinder(MeshCat.Point3f(-0.05,0,0), MeshCat.Point3f(0.05,0,0), 0.05f0)  # 关节
    dim1  = MeshCat.Vec(thick, thick, l1)
    link1 = MeshCat.HyperRectangle(MeshCat.Vec(-thick/2,-thick/2,0), dim1)  # 第一连杆
    dim2  = MeshCat.Vec(thick, thick, l2)
    link2 = MeshCat.HyperRectangle(MeshCat.Vec(-thick/2,-thick/2,0), dim2)  # 第二连杆
    
    # 材质设置
    mat1 = MeshPhongMaterial(color=colorant"grey")  # 关节材质
    mat2 = MeshPhongMaterial(color=color)           # 连杆材质
    
    # 设置几何对象
    setobject!(vis["base"], hinge, mat1)                    # 基座关节
    setobject!(vis["link1"], link1, mat2)                   # 第一连杆
    setobject!(vis["link1","joint"], hinge, mat1)           # 第二关节
    setobject!(vis["link1","link2"], link2, mat2)           # 第二连杆
    
    # 设置变换
    settransform!(vis["link1","link2"], MeshCat.Translation(0,0,l1))
    settransform!(vis["link1","joint"], MeshCat.Translation(0,0,l1))
end

function RotX(alpha)
    """绕X轴旋转矩阵"""
    c, s = cos(alpha), sin(alpha)
    [1 0 0; 0 c -s; 0 s  c]
end

function update_acro_pose!(vis, x)
    """
    更新Acrobot的姿态
    
    参数：
    - vis: 可视化器
    - x: 当前状态 [θ1, θ2, θ1dot, θ2dot]
    """
    l1, l2 = [1, 1.]
    # 第一连杆变换（考虑坐标系偏移π/2）
    settransform!(vis["robot","link1"], MeshCat.LinearMap(RotX(x[1]-pi/2)))
    # 第二连杆变换（相对于第一连杆）
    settransform!(vis["robot","link1","link2"], 
                  MeshCat.compose(MeshCat.Translation(0,0,l1), MeshCat.LinearMap(RotX(x[2]))))
end

# 创建可视化器和动画
vis = Visualizer()
build_acrobot!(vis["robot"])
anim = MeshCat.Animation(vis; fps=floor(Int, 1.0/h))

# 为每个时间步创建动画帧
for k = 1:Nt
    MeshCat.atframe(anim, k) do
        update_acro_pose!(vis, xtraj[:,k])
    end
end

# 设置并渲染动画
MeshCat.setanimation!(vis, anim)
render(vis)