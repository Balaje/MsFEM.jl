using pLOD2d
using StaticArrays
using Gridap

## Parameters

T₁ = Float64;
domain = @SVector T₁[0,1,0,1];

## Fine scale discretization

n = 128;
model_fine = CartesianDiscreteModel(domain, (n,n));
reffe = ReferenceFE(lagrangian, T₁, 1);
V = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁});
Ω = get_triangulation(V);
dΩ = Measure(Ω, 4);

## Diffusion Coefficient

epsilon = min(2^6, n)
repeat_dims = (Int64(n/epsilon), Int64(n/epsilon))
a₁,b₁ = T₁.((0.1,1.0))
using Random
Random.seed!(1234); 
rand_vals = rand(T₁,epsilon^2)
vals_epsilon = repeat(reshape(a₁ .+ (b₁-a₁)*rand_vals, (epsilon, epsilon)), inner=repeat_dims)
A = CellField(vec(vals_epsilon), Ω);

## Multiscale bases

N = 8;
l = 1;
p = 1;

aₕ(u,v) = ∫(A*∇(u)⋅∇(v))dΩ;

β = multiscale_bases(aₕ, V, domain, n, N, l, p);
γ = stabilized_multiscale_bases(aₕ, V, domain, n, N, l, p);
δ = additional_correction_bases(γ, 1, aₕ, V, domain, n, N, l, p);