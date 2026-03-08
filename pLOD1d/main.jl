using pLOD1d
using StaticArrays
using Gridap

N = 4;
n = 2048;
l = 1;
p = 1;

T₁ = Float64;

domain = @SVector T₁[0,1];
model_fine, model_coarse = generate_triangulations(domain, n, N);
reffe = ReferenceFE(lagrangian, T₁, 1);
# Fine scale space.
V = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁});
Ω = get_triangulation(V);
dΩ = Measure(Ω, 6);

D(x) = (1.0 + 0.5*cos(2π*x[1]/2^-6))^-1 # Oscillatory diffusion
f(x) = π^2*sin(π*x[1])

aₕ(u,v) = ∫(D*∇(u)⋅∇(v))dΩ;
lₕ(v) = ∫(f*v)dΩ;
mₕ(u,v) = ∫(u*v)dΩ;

V₀ = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁}, dirichlet_tags=["boundary"]);
op = AffineFEOperator(aₕ, lₕ, V₀, V₀)
uₑ = FEFunction(V₀, op.op.matrix\op.op.vector);


Kₑ, fₑ = assemble_matrix_and_vector(aₕ, lₕ, V, V);
β = reduce(hcat, multiscale_basis(aₕ, V, domain, n, N, l, p));
γ = reduce(hcat, stabilized_multiscale_bases(aₕ, V, domain, n, N, l, p));

function solve_ms_problem(β::AbstractMatrix{T}) where T
  Kₘₛ = β'*Kₑ*β;
  fₘₛ = β'*fₑ;
  uₘₛ = Kₘₛ\fₘₛ;
  β*uₘₛ;
end

uₕ₁ = FEFunction(V, solve_ms_problem(β));
uₕ₂ = FEFunction(V, solve_ms_problem(γ));

e₁ = uₑ - uₕ₁
e₂ = uₑ - uₕ₂

h1err₁ = √(∑(aₕ(e₁, e₁))); l2err₁ = √(∑(mₕ(e₁, e₁)))
h1err₂ = √(∑(aₕ(e₂, e₂))); l2err₂ = √(∑(mₕ(e₂, e₂)))

println("$n \t $N \t $p \t $l \t $l2err₁ \t $h1err₁")
println("$n \t $N \t $p \t $l \t $l2err₂ \t $h1err₂")