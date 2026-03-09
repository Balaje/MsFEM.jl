using pLOD1d
using StaticArrays
using Gridap

T₁ = Float64;
domain = @SVector T₁[0,1];

## Fine-scale Discretization

n = 2048;
model_fine = CartesianDiscreteModel(domain, (n,));
reffe = ReferenceFE(lagrangian, T₁, 1);
V = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁}); # Fine scale space
Ω = get_triangulation(V);
dΩ = Measure(Ω, 4);

## Diffusion coefficient

epsilon = min(256, n)
repeat_dims = Int(n/epsilon)
a₁,b₁ = T₁.((0.1,1.0))
using Random
Random.seed!(1234); 
rand_vals = rand(T₁,epsilon)
vals_epsilon = repeat(a₁ .+ (b₁-a₁)*rand_vals, inner=repeat_dims)
A = CellField(vec(vals_epsilon), Ω);

## Bilinear and linear forms

f(x) = T₁(cos(π*x[1]));
aₕ(u,v) = ∫(A*∇(u)⋅∇(v))dΩ;
lₕ(v) = ∫(f*v)dΩ;
mₕ(u,v) = ∫(u*v)dΩ;

## Compute reference solution

V₀ = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁}, dirichlet_tags=["boundary"]);
op = AffineFEOperator(aₕ, lₕ, V₀, V₀);
uₑ = FEFunction(V₀, op.op.matrix\op.op.vector);

## Compute multiscale solution
N = 8;
p = 1;
l = 1;

Kₑ, fₑ = assemble_matrix_and_vector(aₕ, lₕ, V, V);

"""
Function to solve the Poisson problem given a basis
"""
function solve_poisson_ms(β::Vector{Matrix{T}}) where T<:Real
  Bₘₛ = reduce(hcat, β)
  Kₘₛ = Bₘₛ'*Kₑ*Bₘₛ;
  fₘₛ = Bₘₛ'*fₑ;
  uₘₛ = Kₘₛ\fₘₛ;
  Bₘₛ*uₘₛ;
end;

β = multiscale_bases(aₕ, V, domain, n, N, l, p);
γ = stabilized_multiscale_bases(aₕ, V, domain, n, N, l, p);

uₕ₁ = FEFunction(V, solve_poisson_ms(β));
uₕ₂ = FEFunction(V, solve_poisson_ms(γ));

e₁ = uₕ₁ - uₑ;
e₂ = uₕ₂ - uₑ;

println("$n \t $N \t $p \t $l \t $(√(∑(mₕ(e₁,e₁)))) \t $(√(∑(aₕ(e₁,e₁)))) \t $(√(∑(mₕ(e₂,e₂)))) \t $(√(∑(aₕ(e₂,e₂))))")