using pLOD1d
using StaticArrays
using Gridap

T₁ = Float64;

## Data
domain = @SVector T₁[0,1];
D(x) = (1.0 + 0.5*cos(2π*x[1]/2^-6))^-1 # Oscillatory diffusion

model_fine = CartesianDiscreteModel(domain, (n,));
reffe = ReferenceFE(lagrangian, T₁, 1);

## Fine scale space

n = 2048;
V = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁});
Ω = get_triangulation(V);
dΩ = Measure(Ω, 6);

## Multiscale Space
N = 4;
p = 1;
l = 1;

aₕ(u,v) = ∫(D*∇(u)⋅∇(v))dΩ;

α = multiscale_bases(aₕ, V, domain, n, N, l, p);
β = stabilized_multiscale_bases(aₕ, V, domain, n, N, l, p);
γᵦ = additional_correction_bases(β, j, aₕ, V, domain, n, N, l, p);