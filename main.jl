include("./triangulations.jl");
include("./legendre.jl");
include("./multiscale_bases.jl");
include("./stabilization.jl");

N = 16;
n = 128;
l = 2;
p = 1;

T₁ = Float64;

domain = @SVector T₁[0,1,0,1];
model_fine, model_coarse = generate_triangulations(domain, n, N);
reffe = ReferenceFE(lagrangian, T₁, 1);
# Fine scale space.
V = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁});
Ω = get_triangulation(V);
dΩ = Measure(Ω, 4);

epsilon = min(2^6, n)
repeat_dims = (Int64(n/epsilon), Int64(n/epsilon))
a₁,b₁ = T₁.((0.1,1.0))
using Random
Random.seed!(1234); 
rand_vals = rand(T₁,epsilon^2)
vals_epsilon = repeat(reshape(a₁ .+ (b₁-a₁)*rand_vals, (epsilon, epsilon)), inner=repeat_dims)
A = CellField(vec(vals_epsilon), Ω);

# A = CellField(ones(n*n), Ω)
f(x) = 2π^2*sin(π*x[1])*sin(π*x[2]);
aₕ(u,v) = ∫(A*∇(u)⋅∇(v))dΩ;
lₕ(v) = ∫(f*v)dΩ;

# Compute reference solution
V₀ = FESpace(model_fine, reffe, conformity=:H1, vector_type=Vector{T₁}, dirichlet_tags=["boundary"]);
op = AffineFEOperator(aₕ, lₕ, V₀, V₀);
uₑ = solve(op)

β = reduce(hcat, multiscale_basis(aₕ, V, domain, n, N, l, p));
# β = reduce(hcat, stabilized_multiscale_bases(aₕ, V, domain, n, N, l, p));

Kₑ, fₑ = assemble_matrix_and_vector(aₕ, lₕ, V, V);

Kₘₛ = β'*Kₑ*β;
fₘₛ = β'*fₑ;
uₘₛ = Kₘₛ\fₘₛ;
u = β*uₘₛ;

uₕ = FEFunction(V, u);

e = uₑ - uₕ
@show √(∑(∫(e*e)dΩ)), √(∑(aₕ(e,e)))