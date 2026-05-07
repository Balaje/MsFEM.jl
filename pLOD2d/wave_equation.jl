using pLOD2d
using StaticArrays
using Gridap
using DelimitedFiles

T₁ = Float64
parsed_args = parse_command_line()
n = parsed_args["fine_scale"]
N = parsed_args["coarse_scale"]
p = parsed_args["order"]
l = parsed_args["patch_radius"]
j = parsed_args["correction_level"]
ref_sol = parsed_args["reference_sol"]

## Problem data

domain = @SVector T₁[0,1,0,1];
f(x,t) = T₁(sin(π*x[1])*sin(π*x[2])*sin(t)^7);
u₀(x) = 0.0;
uₜ₀(x) = 0.0;
tf = 1.0;

## Fine Scale Discretization

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

include("./set_solver.jl");
using LinearAlgebra, LinearMaps, LinearSolve

function get_sol(u)
  n = length(u) ÷ 2
  u[n+1:end]
end;

# Time Discretization
dt = 2^-9;
tspan = (0.0, tf);

## Compute the reference solution

V₀ = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁}, dirichlet_tags=["boundary"]); # Reference solution space

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

if(ref_sol == "")
  M = assemble_matrix(mₕ, V₀, V₀);
  K = assemble_matrix(aₕ, V₀, V₀);
  
  M⁻¹ = InverseMap(M; solver=solver)
  U₀ = M⁻¹*assemble_vector(v->∫(u₀*v)dΩ, V₀);
  Uₜ₀ = M⁻¹*assemble_vector(v->∫(uₜ₀*v)dΩ, V₀);
  g(t) = assemble_vector(v->lₕ(v,t), V₀)
  
  ode_solver = RKN4()
  ode_prob = set_solver(M, K, g, U₀, Uₜ₀, tspan, ode_solver)

  s = OrdinaryDiffEq.solve(ode_prob, ode_solver, dt = dt, 
                          save_start=false, save_everystep=false, save_end=true, 
                          progress=true, progress_steps=1, progress_name="Reference solution",
                          adaptive=false);
  
  U = get_sol(s.u[end]);

  using Dates
  open("./ref_sol_n$(n)_T$(Dates.format(now(), "yyyymmddHHMMSS")).txt", "w") do io
    writedlm(io, U)
  end
else
    U = readdlm(ref_sol, T₁)
end

uₑ = FEFunction(V₀, vec(U));

## Compute the Multiscale solution

V = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁}); # Fine scale space
Mₑ = assemble_matrix(mₕ, V, V);
Kₑ = assemble_matrix(aₕ, V, V);

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
  U₀ₘₛ = Mₘₛ⁻¹*(Bₘₛ'*assemble_vector(v->∫(u₀*v)dΩ, V))
  Uₜ₀ₘₛ = Mₘₛ⁻¹*(Bₘₛ'*assemble_vector(v->∫(uₜ₀*v)dΩ, V))

  g(t) = Bₘₛ'*assemble_vector(v->lₕ(v,t), V)
  
  ode_solver = RadauIIA5(linsolve=LUFactorization())
  ode_prob = set_solver(Mₘₛ, Kₘₛ, g, U₀ₘₛ, Uₜ₀ₘₛ, tspan, ode_solver)
  s = OrdinaryDiffEq.solve(ode_prob, ode_solver, dt = dt, 
                           save_start=false, save_everystep=false, save_end=true, 
                           progress=true, progress_steps=1, progress_name="Multiscale solution",
                           adaptive=false);
  
  Uₘₛ = get_sol(s.u[end]);
  
  FEFunction(V, Bₘₛ*Uₘₛ);
end;

uₘₛ = solve_wave_equation_ms(β, j);
e = uₑ - uₘₛ;

println("$n \t $N \t $p \t $l \t $j \t $(√(∑(mₕ(e,e)))) \t $(√(∑(aₕ(e,e))))")