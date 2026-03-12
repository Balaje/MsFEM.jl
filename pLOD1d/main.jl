using pLOD1d
using StaticArrays
using Gridap

T₁ = Float64;

parsed_args = parse_command_line()
n = parsed_args["fine_scale"]
N = parsed_args["coarse_scale"]
p = parsed_args["order"]
l = parsed_args["patch_radius"]
j = parsed_args["correction_level"]

## Data
domain = @SVector T₁[0,1];
D(x) = (1.0 + 0.5*cos(2π*x[1]/2^-6))^-1 # Oscillatory diffusion

## Fine scale space

model_fine = CartesianDiscreteModel(domain, (n,));
reffe = ReferenceFE(lagrangian, T₁, 1);
V = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁});
Ω = get_triangulation(V);
dΩ = Measure(Ω, 6);

## Bilinear forms
aₕ(u,v) = ∫(D*∇(u)⋅∇(v))dΩ;

## Multiscale Space
α = multiscale_bases(aₕ, V, domain, n, N, l, p);
β = stabilized_multiscale_bases(aₕ, V, domain, n, N, l, p);
γᵦ = additional_correction_bases(β, j, aₕ, V, domain, n, N, l, p);