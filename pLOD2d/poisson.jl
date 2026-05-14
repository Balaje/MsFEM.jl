using pLOD2d
using StaticArrays
using Gridap

T₁ = Float64;

parsed_args = parse_command_line()
n = parsed_args["fine_scale"]
N = parsed_args["coarse_scale"]
p = parsed_args["order"]
l = parsed_args["patch_radius"]
j = parsed_args["correction_level"]

## Fine scale discretization

domain = @SVector T₁[0,1,0,1];
model_fine = CartesianDiscreteModel(domain, (n,n));
reffe = ReferenceFE(lagrangian, T₁, 1);
Ω = Triangulation(model_fine)
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

## Problem data

f(x) = T₁((x[1] + cos(3π*x[1]))*x[2]^3);
aₕ(u,v) = ∫(A*∇(u)⋅∇(v))dΩ;
lₕ(v) = ∫(f*v)dΩ;
mₕ(u,v) = ∫(u*v)dΩ;

## Compute reference solution

V₀ = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁}, dirichlet_tags=["boundary"]);
op = AffineFEOperator(aₕ, lₕ, V₀, V₀);
uₑ = FEFunction(V₀, op.op.matrix\op.op.vector);

## Solve the multiscale problem

V = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁});
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

using PrettyTables

β = multiscale_bases(aₕ, V, domain, n, N, l, p; show_progress=false);
γ = stabilized_multiscale_bases(aₕ, V, domain, n, N, l, p; strategy=HLM25(), show_progress=false);
δ = stabilized_multiscale_bases(aₕ, V, domain, n, N, l, p; strategy=DHM25(), show_progress=false);

uₕ₁ = FEFunction(V, solve_poisson_ms(β));
uₕ₂ = FEFunction(V, solve_poisson_ms(γ));
uₕ₃ = FEFunction(V, solve_poisson_ms(δ));

e₁ = uₕ₁ - uₑ;
e₂ = uₕ₂ - uₑ;
e₃ = uₕ₃ - uₑ;

d = ["$n" "$N" "$p" "$l" "$(√(∑(mₕ(e₁,e₁))))" "$(√(∑(aₕ(e₁,e₁))))" "$(√(∑(mₕ(e₂,e₂))))" "$(√(∑(aₕ(e₂,e₂))))" "$(√(∑(mₕ(e₃,e₃))))" "$(√(∑(aₕ(e₃,e₃))))"];
c_labels = ["1/h", "1/H", "p", "l", "L²(pLOD)", "Energy(pLOD)", "L²(spLOD, [HLM25])", "Energy(spLOD, [HLM25])",  "L²(spLOD, [DHM25])", "Energy(spLOD, [DHM25])"]

fname = parsed_args["output_file"]
if(fname=="")
  pretty_table(d; column_labels=c_labels)
else
  open("poisson2d-output-$fname.txt", "w") do io
    pretty_table(io, d; column_labels=c_labels)
  end
end