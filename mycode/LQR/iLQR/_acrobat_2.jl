# 激活当前目录的项目环境并安装依赖包
import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()

# 导入所需的包
using LinearAlgebra      # 线性代数运算
using PyPlot            # 绘图库
using ForwardDiff       # 自动微分，用于计算雅可比矩阵和梯度
using RobotZoo          # 机器人动力学模型库
using RobotDynamics     # 机器人动力学计算
using MatrixCalculus    # 矩阵微积分
using JLD2              # 数据保存和加载

#=================== Acrobot动力学模型定义 ===================#
# 创建Acrobot（双摆）模型实例
a = RobotZoo.Acrobot()
h = 0.05  # 时间步长（秒）

"""
使用四阶龙格-库塔(RK4)方法进行数值积分
这是一种高精度的数值积分方法，用于求解微分方程

参数:
- x: 当前状态向量 [θ₁, θ₂, θ̇₁, θ̇₂]
- u: 控制输入（施加在第二个关节的力矩）

返回:
- 下一时刻的状态向量

RK4积分公式：
x_{k+1} = x_k + (h/6)(f₁ + 2f₂ + 2f₃ + f₄)
其中f₁,f₂,f₃,f₄是不同时刻的导数
"""
function dynamics_rk4(x,u)
    # RK4积分，对控制输入u采用零阶保持
    f1 = RobotZoo.dynamics(a, x, u)                    # k₁ = f(x_k, u_k)
    f2 = RobotZoo.dynamics(a, x + 0.5*h*f1, u)        # k₂ = f(x_k + h/2·k₁, u_k)
    f3 = RobotZoo.dynamics(a, x + 0.5*h*f2, u)        # k₃ = f(x_k + h/2·k₂, u_k)
    f4 = RobotZoo.dynamics(a, x + h*f3, u)            # k₄ = f(x_k + h·k₃, u_k)
    return x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)       # RK4更新公式
end

#=================== 雅可比矩阵计算函数 ===================#
# 这些函数用于计算动力学函数相对于状态和控制输入的各阶导数
# 在DDP算法中需要用到这些导数来构建二次近似

"""
计算动力学函数关于状态x的雅可比矩阵 A = ∂f/∂x
这是线性化后的系统矩阵，用于局部线性近似
"""
function dfdx(x,u)
    ForwardDiff.jacobian(dx->dynamics_rk4(dx,u),x)
end

"""
计算动力学函数关于控制输入u的雅可比矩阵 B = ∂f/∂u
这是控制矩阵，描述控制输入如何影响状态变化
"""
function dfdu(x,u)
    ForwardDiff.derivative(du->dynamics_rk4(x,du),u)
end

"""
计算A矩阵关于状态x的雅可比矩阵 ∂A/∂x = ∂²f/∂x²
这是Hessian矩阵，用于DDP算法的完整牛顿法版本
"""
function dAdx(x,u)
    ForwardDiff.jacobian(dx->vec(dfdx(dx,u)),x)
end

"""
计算B矩阵关于状态x的雅可比矩阵 ∂B/∂x = ∂²f/∂x∂u
这是混合二阶导数
"""
function dBdx(x,u)
    ForwardDiff.jacobian(dx->dfdu(dx,u),x)
end

"""
计算A矩阵关于控制输入u的导数 ∂A/∂u = ∂²f/∂x∂u
"""
function dAdu(x,u)
    ForwardDiff.derivative(du->vec(dfdx(x,du)),u)
end

"""
计算B矩阵关于控制输入u的导数 ∂B/∂u = ∂²f/∂u²
"""
function dBdu(x,u)
    ForwardDiff.derivative(du->dfdu(x,du),u)
end

#=================== 问题参数设置 ===================#
Nx = 4         # 状态维数：[θ₁, θ₂, θ̇₁, θ̇₂]
Nu = 1         # 控制输入维数：施加在关节2的力矩
Tfinal = 10.0  # 总时间（秒）
Nt = Int(Tfinal/h)+1    # 时间步数
thist = Array(range(0,h*(Nt-1), step=h));  # 时间网格

#=================== 代价函数权重设置 ===================#
# 状态代价权重矩阵Q：对位置和速度采用不同权重
# 位置偏差的权重较大(1.0)，速度偏差的权重较小(0.1)
Q = Diagonal([1.0*ones(2); 0.1*ones(2)]);
R = 0.01;          # 控制输入代价权重（惩罚过大的控制力矩）
Qn = Array(100.0*I(Nx));  # 终端代价权重矩阵（比阶段代价权重大，确保终端状态准确）

"""
阶段代价函数：l(x,u) = ½(x-x_goal)ᵀQ(x-x_goal) + ½Ru²
衡量当前状态偏离目标状态的代价和控制输入的代价
"""
function stage_cost(x,u)
    return 0.5*((x-xgoal)'*Q*(x-xgoal)) + 0.5*R*u*u
end

"""
终端代价函数：φ(x_N) = ½(x_N-x_goal)ᵀQ_N(x_N-x_goal)
衡量最终状态偏离目标状态的代价
"""
function terminal_cost(x)
    return 0.5*(x-xgoal)'*Qn*(x-xgoal)
end

"""
总代价函数：J = Σᵢ₌₀^{N-1} l(xᵢ,uᵢ) + φ(x_N)
这是最优控制要最小化的目标函数
"""
function cost(xtraj,utraj)
    J = 0.0
    # 累加所有阶段代价
    for k = 1:(Nt-1)
        J += stage_cost(xtraj[:,k],utraj[k])
    end
    # 加上终端代价
    J += terminal_cost(xtraj[:,Nt])
    return J
end

#=================== DDP算法的反向传播步骤 ===================#
"""
反向传播计算值函数的梯度和Hessian矩阵

这是DDP算法的核心部分，从终端时刻开始向前递推，计算：
- p: 值函数的梯度 ∇V(x)
- P: 值函数的Hessian矩阵 ∇²V(x)  
- d: 前馈控制修正项
- K: 反馈增益矩阵

参数:
- p, P, d, K: 输出数组，存储计算结果
- 这些数组会在函数中被修改

返回:
- ΔJ: 预期的代价函数减少量
"""
function backward_pass!(p,P,d,K)
    
    ΔJ = 0.0
    # 终端边界条件：V_N(x) = φ(x) = ½(x-x_goal)ᵀQ_N(x-x_goal)
    p[:,Nt] .= Qn*(xtraj[:,Nt]-xgoal)     # ∇V_N = Q_N(x_N - x_goal)
    P[:,:,Nt] .= Qn                       # ∇²V_N = Q_N
    
    # 从终端时刻向前递推
    for k = (Nt-1):-1:1
        #=============== 计算代价函数的导数 ===============#
        q = Q*(xtraj[:,k]-xgoal)          # ∇_x l(x,u) = Q(x-x_goal)
        r = R*utraj[k]                    # ∇_u l(x,u) = Ru
    
        # 计算动力学函数的雅可比矩阵（线性化）
        A = dfdx(xtraj[:,k], utraj[k])    # ∂f/∂x
        B = dfdu(xtraj[:,k], utraj[k])    # ∂f/∂u
    
        #=============== 计算Q函数的梯度 ===============#
        # Q函数：Q(x,u) = l(x,u) + V(f(x,u))
        gx = q + A'*p[:,k+1]              # ∇_x Q = ∇_x l + A^T ∇V_{k+1}
        gu = r + B'*p[:,k+1]              # ∇_u Q = ∇_u l + B^T ∇V_{k+1}
    
        #=============== 计算Q函数的Hessian矩阵 ===============#
        # 这里使用iLQR(Gauss-Newton)近似，忽略二阶导数项
        Gxx = Q + A'*P[:,:,k+1]*A         # ∇²_{xx} Q ≈ Q + A^T P_{k+1} A
        Guu = R + B'*P[:,:,k+1]*B         # ∇²_{uu} Q ≈ R + B^T P_{k+1} B  
        Gxu = A'*P[:,:,k+1]*B             # ∇²_{xu} Q ≈ A^T P_{k+1} B
        Gux = B'*P[:,:,k+1]*A             # ∇²_{ux} Q ≈ B^T P_{k+1} A
        
        #=============== DDP完整牛顿法版本（已注释） ===============#
        # 如果要使用完整的牛顿法，需要包含二阶导数项
        # 这会提供更准确的Hessian近似，但计算成本更高
        #Ax = dAdx(xtraj[:,k], utraj[k])
        #Bx = dBdx(xtraj[:,k], utraj[k])
        #Au = dAdu(xtraj[:,k], utraj[k])
        #Bu = dBdu(xtraj[:,k], utraj[k])
        #Gxx = Q + A'*P[:,:,k+1]*A + kron(p[:,k+1]',I(Nx))*comm(Nx,Nx)*Ax
        #Guu = R + B'*P[:,:,k+1]*B + (kron(p[:,k+1]',I(Nu))*comm(Nx,Nu)*Bu)[1]
        #Gxu = A'*P[:,:,k+1]*B + kron(p[:,k+1]',I(Nx))*comm(Nx,Nx)*Au
        #Gux = B'*P[:,:,k+1]*A + kron(p[:,k+1]',I(Nu))*comm(Nx,Nu)*Bx
        
        #=============== 正则化（已注释） ===============#
        # 如果Hessian矩阵不正定，可以添加正则化项
        #β = 0.1
        #while !isposdef(Symmetric([Gxx Gxu; Gux Guu]))
        #    Gxx += β*I
        #    Guu += β*I
        #    β = 2*β
        #    display("regularizing G")
        #    display(β)
        #end
        
        #=============== 计算最优控制策略 ===============#
        # 前馈项：δu* = -G_{uu}^{-1} ∇_u Q
        d[k] = Guu\gu
        # 反馈增益：K = -G_{uu}^{-1} G_{ux}  
        K[:,:,k] .= Guu\Gux
    
        #=============== 更新值函数参数 ===============#
        # 值函数的梯度更新
        p[:,k] = gx - K[:,:,k]'*gu + K[:,:,k]'*Guu*d[k] - Gxu*d[k]
        # 值函数的Hessian更新
        P[:,:,k] .= Gxx + K[:,:,k]'*Guu*K[:,:,k] - Gxu*K[:,:,k] - K[:,:,k]'*Gux
    
        # 累积预期的代价函数改进量
        ΔJ += gu'*d[k]
    end
    
    return ΔJ
end

#=================== 初始化和初始猜测 ===================#
# 初始状态：下垂位置 [-π/2, 0, 0, 0]
x0 = [-pi/2; 0; 0; 0]
# 目标状态：倒立位置 [π/2, 0, 0, 0]  
xgoal = [pi/2; 0; 0; 0]
# 初始轨迹猜测：所有时刻都是初始状态
xtraj = kron(ones(1,Nt), x0)
# 初始控制序列：随机生成
utraj = randn(Nt-1);
# 或者从文件加载之前保存的猜测
#f = jldopen("guess.jld2", "r")
#utraj = f["utraj"];

#=================== 初始前向积分 ===================#
# 使用初始控制序列进行前向积分，得到对应的状态轨迹
for k = 1:(Nt-1)
    xtraj[:,k+1] .= dynamics_rk4(xtraj[:,k],utraj[k])
end
J = cost(xtraj,utraj)  # 计算初始代价

#=================== DDP算法主循环 ===================#
using Printf

# 初始化DDP算法所需的数组
p = ones(Nx,Nt)           # 值函数梯度
P = zeros(Nx,Nx,Nt)       # 值函数Hessian矩阵
d = ones(Nt-1)            # 前馈控制修正
K = zeros(Nu,Nx,Nt-1)     # 反馈增益矩阵
ΔJ = 0.0                  # 预期代价改进量

# 临时存储新的轨迹
xn = zeros(Nx,Nt)         # 新的状态轨迹
un = zeros(Nt-1)          # 新的控制轨迹

# 临时存储梯度和Hessian
gx = zeros(Nx)
gu = 0.0
Gxx = zeros(Nx,Nx)
Guu = 0.0
Gxu = zeros(Nx)
Gux = zeros(Nx)

# DDP迭代主循环
iter = 0
# 收敛条件：前馈修正项的最大绝对值小于阈值
while maximum(abs.(d[:])) >  1e-3
    iter += 1    
    
    #=============== 反向传播步骤 ===============#
    # 计算值函数参数和控制策略
    ΔJ = backward_pass!(p,P,d,K)

    #=============== 前向积分与线搜索 ===============#
    # 使用新的控制策略进行前向积分
    xn[:,1] = xtraj[:,1]   # 初始状态保持不变
    α = 1.0                # 线搜索步长
    
    # 应用新的控制策略：u_new = u_old - α*d - K*(x_new - x_old)
    for k = 1:(Nt-1)
        un[k] = utraj[k] - α*d[k] - dot(K[:,:,k],xn[:,k]-xtraj[:,k])
        xn[:,k+1] .= dynamics_rk4(xn[:,k],un[k])
    end
    Jn = cost(xn,un)  # 计算新轨迹的代价
    
    #=============== Armijo线搜索 ===============#
    # 如果新轨迹的代价没有足够改善，缩小步长重试
    # Armijo条件：J_new ≤ J_old - c₁*α*ΔJ，这里c₁ = 1e-2
    while isnan(Jn) || Jn > (J - 1e-2*α*ΔJ)
        α = 0.5*α  # 步长减半
        for k = 1:(Nt-1)
            un[k] = utraj[k] - α*d[k] - dot(K[:,:,k],xn[:,k]-xtraj[:,k])
            xn[:,k+1] .= dynamics_rk4(xn[:,k],un[k])
        end
        Jn = cost(xn,un)
    end

    #=============== 算法状态输出 ===============#
    # 每100次迭代输出表头
    if rem(iter - 1, 100) == 0
        @printf "iter     J           ΔJ        |d|         α       \n"
        @printf "---------------------------------------------------\n"
    end
    # 每10次迭代输出当前状态
    if rem(iter - 1, 10) == 0 
        @printf("%3d   %10.3e  %9.2e  %9.2e  %6.4f  \n",
              iter, J, ΔJ, maximum(abs.(d[:])), α)
    end
    
    #=============== 更新轨迹 ===============#
    J = Jn           # 更新代价
    xtraj .= xn      # 更新状态轨迹
    utraj .= un      # 更新控制轨迹
end

#=================== 结果可视化 ===================#
# 绘制状态轨迹
plot(thist,xtraj[1,:])  # 第一个关节角度
plot(thist,xtraj[2,:])  # 第二个关节角度

# 绘制控制输入
plot(thist[1:Nt-1],utraj)

#=================== 3D动画可视化 ===================#
# 使用MeshCat库创建3D动画
import MeshCat as mc
using Colors

"""
绕X轴旋转的旋转矩阵
"""
function RotX(alpha)
    c, s = cos(alpha), sin(alpha)
    [1 0 0; 0 c -s; 0 s  c]
end

"""
创建Acrobot的3D模型
- vis: MeshCat可视化器
- color: 连杆颜色
- thick: 连杆厚度
"""
function create_acrobot!(vis, color=colorant"blue", thick=0.05)
    l1,l2 = [1.,1.]  # 两个连杆的长度
    
    # 创建关节（圆柱体）
    hinge = mc.Cylinder(mc.Point(-0.05,0,0), mc.Point(0.05,0,0), 0.05)
    
    # 创建第一个连杆（长方体）
    dim1  = mc.Vec(thick, thick, l1)
    link1 = mc.HyperRectangle(mc.Vec(-thick/2,-thick/2,0),dim1)
    
    # 创建第二个连杆（长方体）
    dim2  = mc.Vec(thick, thick, l2)
    link2 = mc.HyperRectangle(mc.Vec(-thick/2,-thick/2,0),dim2)
    
    # 设置材质
    mat1 = mc.MeshPhongMaterial(color=colorant"grey")  # 关节颜色
    mat2 = mc.MeshPhongMaterial(color=color)           # 连杆颜色
    
    # 将几何体添加到场景中
    mc.setobject!(vis["base"], hinge, mat1) 
    mc.setobject!(vis["link1"], link1, mat2) 
    mc.setobject!(vis["link1","joint"], hinge, mat1) 
    mc.setobject!(vis["link1","link2"], link2, mat2) 
    
    # 设置第二个连杆和关节的位置
    mc.settransform!(vis["link1","link2"], mc.Translation(0,0,l1))
    mc.settransform!(vis["link1","joint"], mc.Translation(0,0,l1))
end

"""
更新Acrobot的姿态
- vis: 可视化器
- x: 状态向量 [θ₁, θ₂, θ̇₁, θ̇₂]
"""
function update_acro_pose!(vis, x)
    l1, l2 = [1, 1.]
    # 设置第一个连杆的旋转（相对于垂直向下方向）
    mc.settransform!(vis["robot","link1"], mc.LinearMap(RotX(x[1]-pi/2)))
    # 设置第二个连杆的旋转（相对于第一个连杆）
    mc.settransform!(vis["robot","link1","link2"], 
                     mc.compose(mc.Translation(0,0,l1), mc.LinearMap(RotX(x[2]))))
end

"""
创建Acrobot运动的动画
- X: 状态轨迹列表
- dt: 时间步长
"""
function animate_acrobot(X, dt)
    vis = mc.Visualizer()           # 创建可视化器
    create_acrobot!(vis["robot"])   # 创建机器人模型
    anim = mc.Animation(vis; fps=floor(Int,1/dt))  # 创建动画对象
    
    # 为每个时间步创建关键帧
    for k = 1:length(X)
        mc.atframe(anim, k) do
            update_acro_pose!(vis,X[k])  # 更新机器人姿态
        end
    end
    mc.setanimation!(vis, anim)     # 设置动画
    return mc.render(vis)           # 渲染并返回可视化结果
end

# 将状态轨迹转换为向量列表并创建动画
X1 = [Vector(x) for x in eachcol(xtraj)];
animate_acrobot(X1, h)