#### #### #### #### #### #### #### #### #### #### #### 
# Script to set the ODE solver and the linear solver
#### #### #### #### #### #### #### #### #### #### #### 

using OrdinaryDiffEq, OrdinaryDiffEqCore
using OrdinaryDiffEqRKN, OrdinaryDiffEqFIRK, OrdinaryDiffEqRosenbrock

function set_solver(M::AbstractMatrix{T}, K::AbstractMatrix{U}, f::Function, U₀::Vector, Uₜ₀::Vector, tspan::NTuple{2,<:Real}, solver::F)  where {T<:Real, U<:Real, F<:OrdinaryDiffEqCore.OrdinaryDiffEqAlgorithm}
  (solver == RKN4()) ? second_order_solver(M, K, f, U₀, Uₜ₀, tspan) : first_order_solver(M, K, f, U₀, Uₜ₀, tspan)
end;

solver(y,A,b) = y .= A\b;

function second_order_solver(M::AbstractMatrix, K::AbstractMatrix, f::Function, U₀::Vector, Uₜ₀::Vector, tspan::NTuple{2,<:Real})
  M⁻¹ = InverseMap(M, solver=solver)  
  function W(v, u, p, t)    
    -(M⁻¹*K*u) + M⁻¹*f(t)
  end
  SecondOrderODEProblem(W, Uₜ₀, U₀, tspan)
end;

function first_order_solver(M::AbstractMatrix, K::AbstractMatrix, f::Function, U₀::Vector, Uₜ₀::Vector, tspan::NTuple{2,<:Real})
  Mₛ = [M 0*I; 0*I I]
  Kₛ = [0*I -K; I 0*I]
  function W!(du, u, p, t)    
    n = length(u) ÷ 2
    V = @view u[1:n]; 
    U = @view u[n+1:2n]    
    dV = @view du[1:n]; 
    dU = @view du[n+1:2n]  
    dU .= V
    dV .= f(t) - K*U
    return nothing
  end;
  function jac!(J, u, p, t)
    copyto!(J, Kₛ)
    return nothing
  end;
  W = ODEFunction(W!, mass_matrix=Mₛ, jac=jac!, jac_prototype=Kₛ)
  ODEProblem(W, [Uₜ₀; U₀], tspan)  
end