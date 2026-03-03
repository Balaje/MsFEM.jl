using LegendrePolynomials
using LinearAlgebra
using SparseArrays

"""
Function to obtain the exponents of the Legendre polynomials
"""
function poly_exps(p::Int64)
  X = ones(Int64,p+1)*(0:1:p)';
  Y = (0:1:p)*ones(Int64,p+1)';
  vec(map((a,b)->(a,b), X, Y))
end;

"""
Function to generate the Legendre Polynomials on the reference square
"""
function Λ(x::VectorValue{2, T1}, p::NTuple{2,Int}) where T1
  return Pl(x[1], p[1])*Pl(x[2], p[2])  
end;

"""
Function to generate the L-matrix using the reference square [-1,1]²
"""
function Λ(n::Int, N::Int, p::Int)
  a = ceil(Int, n/N);
  model = CartesianDiscreteModel((-1, 1, -1, 1), (a, a))
  V = FESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:H1)
  Ω = get_triangulation(V);
  dΩ = Measure(Ω, 2p+2);
  P = poly_exps(p)
  L = [];
  for pᵢ in P
    λ(x) = Λ(x, pᵢ);
    push!(L, assemble_vector(v->∫(λ*v)dΩ, V))
  end
  h = 1/N;
  jac = (h/2)^2; # Jacobian of the transformation
  reduce(hcat, jac*L), get_cell_node_ids(model)
end;

"""
Function to assemble the global L-matrix.
"""
function assemble_L_matrix(model_fine::CartesianDiscreteModel, n::Int, N::Int, p::Int)
  elem_fine = elements_in_coarse_scale_patch(get_cell_node_ids(model_fine), N, 0);  

  L₁, Z₁ = Λ(n, N, p)
  Z₂ = combinedimsview(vec(Z₁))
  elem_to_dof(i) = (p+1)^2*(i-1)+1:(p+1)^2*i

  X = combinedimsview(combinedimsview.(vec.(elem_fine)))
  Z = map(elem_to_dof, 1:N*N)
  L = spzeros((n+1)*(n+1), N*(p+1)*N*(p+1))
  for i=1:N*N
    L[X[:,:,i], Z[i]] .= L₁[Z₂, :]
  end

  L
end

"""
Function to compute the innerproduct of the LegendrePolynomials
"""
function λ(N::Int, p::Int)
  t = [2/(2*(j-1)+1) for j=1:(p+1)];
  h = 1/N;
  jac = (h/2)^2;
  jac*diagm(repeat(vec(t*t'), outer=N*N));
end;