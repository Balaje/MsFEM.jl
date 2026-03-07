include("./triangulations.jl");
include("./legendre.jl");
include("./multiscale_bases.jl");
include("./stabilization.jl");

n = 128;
p = 1;
l = 1;

lc = :red

using Plots
plt1 = Plots.plot()

T₁ = Float64;
domain = @SVector T₁[0,1,0,1];

model_fine = CartesianDiscreteModel(domain, (n,n));
reffe = ReferenceFE(lagrangian, T₁, 1);
# Reference solution space
V₀ = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁}, dirichlet_tags=["boundary"]);
# Fine scale space
V = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁});
Ω = get_triangulation(V);
dΩ = Measure(Ω, 4);

epsilon = min(64, n)
repeat_dims = (Int64(n/epsilon), Int64(n/epsilon))
a₁,b₁ = T₁.((0.1,1.0))
using Random
Random.seed!(1234); 
rand_vals = rand(T₁,epsilon^2)
vals_epsilon = repeat(reshape(a₁ .+ (b₁-a₁)*rand_vals, (epsilon, epsilon)), inner=repeat_dims)
A = CellField(vec(vals_epsilon), Ω);

f(x) = (x[1] + cos(3π*x[1]))*x[2]^3;
aₕ(u,v) = ∫(A*∇(u)⋅∇(v))dΩ;
lₕ(v) = ∫(f*v)dΩ;

Kₑ, fₑ = assemble_matrix_and_vector(aₕ, lₕ, V, V);

# Compute reference solution
op = AffineFEOperator(aₕ, lₕ, V₀, V₀);
uₑ = Gridap.solve(op)

function solve_ms_problem(β::AbstractMatrix{T}) where T
  Kₘₛ = β'*Kₑ*β;
  fₘₛ = β'*fₑ;
  uₘₛ = Kₘₛ\fₘₛ;
  β*uₘₛ;
end

err₁ = Float64[];
err₂ = Float64[];

Ns = [2,4,8,16]
H = 1 ./ Ns
for N=Ns
  local β = reduce(hcat, multiscale_basis(aₕ, V, domain, n, N, l, p))
  local γ = reduce(hcat, stabilized_multiscale_bases(aₕ, V, domain, n, N, l, p))

  uₕ₁ = FEFunction(V, solve_ms_problem(β));
  uₕ₂ = FEFunction(V, solve_ms_problem(γ));

  e₁ = uₕ₁ - uₑ
  e₂ = uₕ₂ - uₑ

  push!(err₁, √(∑(aₕ(e₁,e₁))))
  push!(err₂, √(∑(aₕ(e₂,e₂))))
  
  println("Done N=$N")
end

Plots.plot!(plt1, H, err₁, xaxis=:log2, yaxis=:log10, label="p-LOD", lc=lc, lw=2, ls=:dash); 
Plots.scatter!(plt1, H, err₁, label="", markershape=:diamond);
Plots.plot!(plt1, H, err₂, xaxis=:log2, yaxis=:log10, label="sp-LOD", lc=lc, lw=2); 
Plots.scatter!(plt1, H, err₂, label="", markershape=:dtriangle);