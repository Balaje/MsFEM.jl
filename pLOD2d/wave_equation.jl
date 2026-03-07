using pLOD2d
using StaticArrays
using Gridap

T₁ = Float32;

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
repeat_dims = (Int64(n/epsilon), Int64(n/epsilon))
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

dt = 2^-9;
tspan = (0.0, tf);

ode_prob = SecondOrderODEProblem(W, Uₜ₀, U₀, tspan, (M⁻¹, K, V₀))
s = OrdinaryDiffEq.solve(ode_prob, ode_solver, dt = dt);

function get_sol(u)
  n = Int64(0.5*length(u))
  u[n+1:2n]
end;

U = get_sol(s.u[end]);

uₑ = FEFunction(V₀, U);

## Compute the Multiscale solution

N = 4;

V = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁}); # Fine scale space

Mₑ = assemble_matrix(mₕ, V, V);
Kₑ = assemble_matrix(aₕ, V, V);

p = 1;
l = 1;
j = 1;

β = stabilized_multiscale_bases(aₕ, V, domain, n, N, l, p);
δ = additional_correction_bases(β, j, aₕ, V, domain, n, N, l, p);

Bₘₛ = reduce(hcat, reduce(hcat, δ));

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
s = OrdinaryDiffEq.solve(ode_prob, ode_solver, dt = dt);

Uₘₛ = get_sol(s.u[end]);

uₘₛ = FEFunction(V, Bₘₛ*Uₘₛ);

e = uₑ - uₘₛ
h1err = √(∑(aₕ(e,e)));
l2err = √(∑(mₕ(e,e)));

println("$n \t $N \t $p \t $l \t $j \t $l2err \t $h1err")