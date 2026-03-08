using pLOD1d
using StaticArrays
using Gridap

n = 2048;
p = 1;
l = 7;

lc = :black

using Plots
# plt1 = Plots.plot()

T₁ = Float64;
domain = @SVector T₁[0,1];

model_fine = CartesianDiscreteModel(domain, (n,));
reffe = ReferenceFE(lagrangian, T₁, 1);
# Reference solution space
V₀ = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁}, dirichlet_tags=["boundary"]);
# Fine scale space
V = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁});
Ω = get_triangulation(V);
dΩ = Measure(Ω, 4);

epsilon = min(64, n)
repeat_dims = Int(n/epsilon)
a₁,b₁ = T₁.((0.1,1.0))
using Random
Random.seed!(1234); 
rand_vals = rand(T₁,epsilon)
vals_epsilon = repeat(a₁ .+ (b₁-a₁)*rand_vals, inner=repeat_dims)
A = CellField(vec(vals_epsilon), Ω);

f(x) = T₁(cos(π*x[1]));
aₕ(u,v) = ∫(A*∇(u)⋅∇(v))dΩ;
lₕ(v) = ∫(f*v)dΩ;

Kₑ, fₑ = assemble_matrix_and_vector(aₕ, lₕ, V, V);

# Compute reference solution
op = AffineFEOperator(aₕ, lₕ, V₀, V₀);
uₑ = FEFunction(V₀, op.op.matrix\op.op.vector);

function solve_ms_problem(β::AbstractMatrix{T}) where T
  Kₘₛ = β'*Kₑ*β;
  fₘₛ = β'*fₑ;
  uₘₛ = Kₘₛ\fₘₛ;
  β*uₘₛ;
end

err₁ = T₁[];
err₂ = T₁[];

Ns = [2,4,8,16,32,64,128]
H = 1 ./ Ns
for N=Ns
  local β = reduce(hcat, multiscale_basis(aₕ, V, domain, n, N, l, p))
  local γ = reduce(hcat, stabilized_multiscale_bases(aₕ, V, domain, n, N, l, p))

  local uₕ₁ = FEFunction(V, solve_ms_problem(β));
  local uₕ₂ = FEFunction(V, solve_ms_problem(γ));

  local e₁ = uₕ₁ - uₑ
  local e₂ = uₕ₂ - uₑ

  push!(err₁, √(∑(aₕ(e₁,e₁))))
  push!(err₂, √(∑(aₕ(e₂,e₂))))
  
  println("Done N=$N")
end

Plots.plot!(plt1, H, err₁, xaxis=:log2, yaxis=:log10, label="p-LOD, l=$l, p=$p", lc=lc, lw=2, ls=:dash); 
Plots.scatter!(plt1, H, err₁, label="", markershape=:diamond);
Plots.plot!(plt1, H, err₂, xaxis=:log2, yaxis=:log10, label="sp-LOD, l=$l, p=$p", lc=lc, lw=2); 
Plots.scatter!(plt1, H, err₂, label="", markershape=:dtriangle);