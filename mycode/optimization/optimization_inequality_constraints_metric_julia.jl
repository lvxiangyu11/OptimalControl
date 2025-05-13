function gauss_newton_step(x, λ)
    H = ∇²f(x)                  # 目标函数 Hessian
    C = ∂c(x)                   # 约束雅可比（行向量）
    # 解 KKT 线性系统 [H  C'; C  0]·[Δx; Δλ] = [-∇f - C'λ; -c]
    Δz = [H  C'; C  0] \ ([-∇f(x) - C' * λ; -c(x)])
    Δx = Δz[1:2];  Δλ = Δz[3]
    return Δx, Δλ
end

# Merit 函数（增广拉格朗日形式）
ρ = 1.0
function P(x, λ)
    f(x) + λ' * c(x) + 0.5 * ρ * dot(c(x), c(x))
end

# Merit 梯度
function ∇P(x, λ)
    g = ∇f(x) + ∂c(x)' * (λ .+ ρ * c(x))
    return [g; c(x)]
end

# 带 Armijo 线搜索的一步更新
function update!(x, λ)
    Δx, Δλ = gauss_newton_step(x, λ)
    α = 1.0
    # Armijo 判据
    while P(x + α * Δx, λ + α * Δλ) > P(x, λ) + 0.01 * α * dot(∇P(x, λ), [Δx; Δλ])
        α *= 0.5
    end
    x_new = x + α * Δx
    λ_new = λ + α * Δλ
    return x_new, λ_new
end
