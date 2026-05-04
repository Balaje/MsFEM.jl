using pLOD1d
using StaticArrays
using Gridap

T₁ = Float64

parsed_args = parse_command_line()
n = parsed_args["fine_scale"]
N = parsed_args["coarse_scale"]
p = parsed_args["order"]
l = parsed_args["patch_radius"]
j = parsed_args["correction_level"]

## Problem data

domain = @SVector T₁[0,1];
f(x,t) = T₁((x[1]+sin(π*x[1]))*sin(t)^7);
u₀(x) = 0.0;
uₜ₀(x) = 0.0; # π*sin(π*x[1]);
tf = 1.0;

## Fine Scale Discretization

model_fine = CartesianDiscreteModel(domain, (n,));
reffe = ReferenceFE(lagrangian, T₁, 1);
Ω = Triangulation(model_fine);
dΩ = Measure(Ω, 4);

## Diffusion Coefficient

epsilon = min(256, n)
repeat_dims = Int64(n/epsilon)
using Random
Random.seed!(1234); 
rand_vals = rand(T₁,epsilon)
a₁,b₁ = T₁.((0.1,1.0))
vals_epsilon = repeat(a₁ .+ (b₁-a₁)*rand_vals, inner=repeat_dims)
A = CellField(vec(vals_epsilon), Ω);

## Weak formulation

aₕ(u,v) = ∫(A*∇(u)⋅∇(v))dΩ;
function lₕ(v,t) 
  g(x) = f(x,t)
  ∫(g*v)dΩ;
end
mₕ(u,v) = ∫(u*v)dΩ;

## ODE Solvers

include("./set_solver.jl");
using LinearAlgebra, LinearMaps, LinearSolve

function get_sol(u)
  n = length(u) ÷ 2
  u[n+1:end]
end;

# Time Discretization
dt = 2^-12;
tspan = (0.0, tf);

## Compute the reference solution

V₀ = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁}, dirichlet_tags=["boundary"]); # Reference solution space
M = assemble_matrix(mₕ, V₀, V₀);
K = assemble_matrix(aₕ, V₀, V₀);

M⁻¹ = InverseMap(M; solver=solver)
U₀ = M⁻¹*assemble_vector(v->∫(u₀*v)dΩ, V₀);
Uₜ₀ = M⁻¹*assemble_vector(v->∫(uₜ₀*v)dΩ, V₀);
g(t) = assemble_vector(v->lₕ(v,t), V₀)

ode_solver = RadauIIA5(linsolve=LUFactorization())
ode_prob = set_solver(M, K, g, U₀, Uₜ₀, tspan, ode_solver)

s = OrdinaryDiffEq.solve(ode_prob, ode_solver, dt = dt, save_start=false, save_everystep=false, save_end=true);

U = get_sol(s.u[end]);

uₑ = FEFunction(V₀, U);

## Compute the Multiscale solution

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
  Mₘₛ⁻¹ = InverseMap(Mₘₛ; solver=solver)
  U₀ₘₛ = Mₘₛ⁻¹*(Bₘₛ'*assemble_vector(v->∫(u₀*v)dΩ, V))
  Uₜ₀ₘₛ = Mₘₛ⁻¹*(Bₘₛ'*assemble_vector(v->∫(uₜ₀*v)dΩ, V))

  g(t) = Bₘₛ'*assemble_vector(v->lₕ(v,t), V)
  
  ode_prob = set_solver(Mₘₛ, Kₘₛ, g, U₀ₘₛ, Uₜ₀ₘₛ, tspan, ode_solver)
  s = OrdinaryDiffEq.solve(ode_prob, ode_solver, dt = dt, save_start=false, save_everystep=false, save_end=true);
  
  Uₘₛ = get_sol(s.u[end]);
  
  FEFunction(V, Bₘₛ*Uₘₛ);
end;

uₘₛ₁ = solve_wave_equation_ms(α, j);
uₘₛ₂ = solve_wave_equation_ms(β, j);
e₁ = uₑ - uₘₛ₁;
e₂ = uₑ - uₘₛ₂;

println("$n \t $N \t $p \t $l \t $j \t $(√(∑(mₕ(e₁,e₁)))) \t $(√(∑(aₕ(e₁,e₁)))) \t $(√(∑(mₕ(e₂,e₂)))) \t $(√(∑(aₕ(e₂,e₂))))")