using pLOD1d
using StaticArrays
using Gridap

T₁ = Float64

## Problem data

domain = @SVector T₁[0,1];
f(x,t) = T₁((x[1]+sin(π*x[1]))*sin(t)^7);
u₀(x) = 0.0;
uₜ₀(x) = 0.0;
tf = 1.0;

## Fine Scale Discretization

n = 2048;
model_fine = CartesianDiscreteModel(domain, (n,));
reffe = ReferenceFE(lagrangian, T₁, 1);
Ω = Triangulation(model_fine);
dΩ = Measure(Ω, 4);

## Diffusion Coefficient

epsilon = min(64, n)
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

dt = 2^-12;
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

using Plots
# plt1 = Plots.plot()
lc = :maroon;

p = 1;
l = 5;
j = 1;

V = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁}); # Fine scale space
Mₑ = assemble_matrix(mₕ, V, V);
Kₑ = assemble_matrix(aₕ, V, V);

l2err₁ = T₁[];  h1err₁ = T₁[];
l2err₂ = T₁[];  h1err₂ = T₁[];

Ns = [2, 4, 8, 16, 32, 64, 128];
H = 1 ./ Ns

for N=Ns  
  α = multiscale_basis(aₕ, V, domain, n, N, l, p);
  β = stabilized_multiscale_bases(aₕ, V, domain, n, N, l, p);  
  
  function solve_wave_ms(β)    
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
    
    FEFunction(V, Bₘₛ*Uₘₛ);
  end;
  
  uₘₛ₁ = solve_wave_ms(α)
  uₘₛ₂ = solve_wave_ms(β)
  e₁ = uₑ - uₘₛ₁
  e₂ = uₑ - uₘₛ₂
  push!(h1err₁, √(∑(aₕ(e₁,e₁))));   push!(l2err₁, √(∑(mₕ(e₁,e₁))))
  push!(h1err₂, √(∑(aₕ(e₂,e₂))));   push!(l2err₂, √(∑(mₕ(e₂,e₂))))
  
  println("$n \t $N \t $p \t $l \t $j \t $(√(∑(mₕ(e₁,e₁)))) \t $(√(∑(aₕ(e₁,e₁)))) \t $(√(∑(mₕ(e₂,e₂)))) \t $(√(∑(aₕ(e₂,e₂))))")
end

Plots.plot!(plt1, H, h1err₁, xaxis=:log2, yaxis=:log10, label="p-LOD, l=$l, p=$p", lc=lc, lw=2, ls=:dash); 
Plots.scatter!(plt1, H, h1err₁, label="", markershape=:diamond);
Plots.plot!(plt1, H, h1err₂, xaxis=:log2, yaxis=:log10, label="sp-LOD, l=$l, p=$p", lc=lc, lw=2); 
Plots.scatter!(plt1, H, h1err₂, label="", markershape=:dtriangle);