using pLOD2d
using StaticArrays
using Gridap

T₁ = Float64

## Problem data

domain = @SVector T₁[0,1,0,1];
f(x,t) = T₁(sin(π*x[1])*sin(π*x[2])*sin(t)^7);
u₀(x) = 0.0;
uₜ₀(x) = 0.0;
tf = 1.0;

## Fine Scale Discretization

n = 128;

model_fine = CartesianDiscreteModel(domain, (n,n));
reffe = ReferenceFE(lagrangian, T₁, 1);
Ω = Triangulation(model_fine);
dΩ = Measure(Ω, 4);

## Diffusion Coefficient

epsilon = min(64, n)
repeat_dims = (Int(n/epsilon), Int(n/epsilon))
a₁,b₁ = T₁.((0.1,1.0))
using Random
Random.seed!(1234); 
rand_vals = rand(T₁,epsilon^2)
vals_epsilon = repeat(reshape(a₁ .+ (b₁-a₁)*rand_vals, (epsilon, epsilon)), inner=repeat_dims)
A = CellField(vec(vals_epsilon), Ω);

## Weak formulation

aₕ(u,v) = ∫(A*∇(u)⋅∇(v))dΩ;
function lₕ(v,t) 
  g(x) = f(x,t)
  ∫(g*v)dΩ;
end
mₕ(u,v) = ∫(u*v)dΩ;

## ODE Solvers

using OrdinaryDiffEqRKN, OrdinaryDiffEq
ode_solver = RKN4()
solver = (y,A,b) -> y .= A\b;

function get_sol(u)
  n = Int64(0.5*length(u))
  u[n+1:2n]
end;

# Time Discretization
dt = 2^-9;
tspan = (0.0, tf);

## Compute the reference solution

V₀ = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁}, dirichlet_tags=["boundary"]); # Reference solution space
M = assemble_matrix(mₕ, V₀, V₀);
K = assemble_matrix(aₕ, V₀, V₀);

using LinearMaps
M⁻¹ = InverseMap(M; solver=solver)
U₀ = M⁻¹*assemble_vector(v->∫(u₀*v)dΩ, V₀);
Uₜ₀ = M⁻¹*assemble_vector(v->∫(uₜ₀*v)dΩ, V₀);

function W(v, u, p, t)
  M⁻¹, K, V = p
  g = assemble_vector(v->lₕ(v,t), V)
  -(M⁻¹*K*u) + M⁻¹*g
end

ode_prob = SecondOrderODEProblem(W, Uₜ₀, U₀, tspan, (M⁻¹, K, V₀))
s = OrdinaryDiffEq.solve(ode_prob, ode_solver, dt = dt);

U = get_sol(s.u[end]);

uₑ = FEFunction(V₀, U);

## Compute the Multiscale solution
N = 8;
p = 3;
l = 4;
j = 2;

V = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁}); # Fine scale space
Mₑ = assemble_matrix(mₕ, V, V);
Kₑ = assemble_matrix(aₕ, V, V);

α = multiscale_bases(aₕ, V, domain, n, N, l, p);
β = stabilized_multiscale_bases(aₕ, V, domain, n, N, l, p);  

"""
Function to solve the Wave Equation given a basis β and the number of additional correction steps
"""
function solve_wave_equation_ms(β::Vector{Matrix{T}}, j::Int) where T<:Real

  # Compute the additional corrections
  γ = additional_correction_bases(β, j, aₕ, V, domain, n, N, l, p);    
  
  Bₘₛ = reduce(hcat, reduce(hcat, γ));
  Kₘₛ = Bₘₛ'*Kₑ*Bₘₛ
  Mₘₛ = Bₘₛ'*Mₑ*Bₘₛ
  Mₘₛ⁻¹ = InverseMap(Mₘₛ; solver=solver);
  
  function Wₘₛ(v, u, p, t)
    Mₘₛ⁻¹, Kₘₛ, V, Bₘₛ  = p
    L = assemble_vector(v->lₕ(v,t), V);
    g = Bₘₛ'*L
    -(Mₘₛ⁻¹*Kₘₛ*u) + Mₘₛ⁻¹*g
  end
  
  U₀ₘₛ = Mₘₛ⁻¹*(Bₘₛ'*assemble_vector(v->∫(u₀*v)dΩ, V))
  Uₜ₀ₘₛ = Mₘₛ⁻¹*(Bₘₛ'*assemble_vector(v->∫(uₜ₀*v)dΩ, V))
  
  ode_prob = SecondOrderODEProblem(Wₘₛ, Uₜ₀ₘₛ, U₀ₘₛ, tspan, (Mₘₛ⁻¹, Kₘₛ, V, Bₘₛ))
  s = OrdinaryDiffEq.solve(ode_prob, ode_solver, dt = dt, 
            save_start=false,
            save_everystep=false,
            save_end=true);
  
  Uₘₛ = get_sol(s.u[end]);
  
  FEFunction(V, Bₘₛ*Uₘₛ);
end;

uₘₛ₁ = solve_wave_equation_ms(α, j);
uₘₛ₂ = solve_wave_equation_ms(β, j);
e₁ = uₑ - uₘₛ₁;
e₂ = uₑ - uₘₛ₂;

println("$n \t $N \t $p \t $l \t $j \t $(√(∑(mₕ(e₁,e₁)))) \t $(√(∑(aₕ(e₁,e₁)))) \t $(√(∑(mₕ(e₂,e₂)))) \t $(√(∑(aₕ(e₂,e₂))))")