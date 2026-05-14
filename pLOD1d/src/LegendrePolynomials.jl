module LegendrePolynomials

using LegendrePolynomials
using LinearAlgebra
using FastGaussQuadrature

using SparseArrays
using StaticArrays
using SplitApplyCombine

using Gridap
using Gridap.Geometry: get_cell_node_ids

using pLOD1d.Triangulations: elements_in_coarse_scale_patch, get_interior

"""
Function to obtain the exponents of the Legendre polynomials
"""
function poly_exps(p::T) where T<:Integer
  X = 0:1:p
  [(x,) for x in X]
end;

"""
Function to generate the Legendre Polynomials on the reference square
"""
function Λ(x::VectorValue{1, T1}, p::NTuple{1,Int}) where T1
  return Pl(x[1], p[1])
end;

"""
Function to generate the rectangular matrix using the reference square [-1,1]²
"""
function reference_rectangular_matrix(domain::SVector{2,T}, n::Int, N::Int, p::Int) where T<:Real
  a = n ÷ N;
  model = CartesianDiscreteModel(domain, (a, ))
  V = FESpace(model, ReferenceFE(lagrangian, T, 1), conformity=:H1; vector_type=Vector{T})
  Ω = get_triangulation(V);
  dΩ = Measure(Ω, 2p+2);
  P = poly_exps(p)
  L = Vector{T}[];
  for pᵢ in P
    λ(x) = Λ(x, pᵢ);
    push!(L, assemble_vector(v->∫(λ*v)dΩ, V))
  end  
  reduce(hcat, L), get_cell_node_ids(model)
end;

"""
Function to assemble the global L-matrix.
"""
function assemble_rectangular_matrix(domain::SVector{2,T}, n::Int, N::Int, p::Int) where T<:Real
  model_fine = CartesianDiscreteModel(domain, (n,))
  elem_fine = elements_in_coarse_scale_patch(get_cell_node_ids(model_fine), N, 0);  

  ref_domain = @SVector T[-1,1]
  L₁, Z₁ = reference_rectangular_matrix(ref_domain, n, N, p)
  Z₂ = combinedimsview(vec(Z₁))
  elem_to_dof(i) = (p+1)*(i-1)+1:(p+1)*i

  X = combinedimsview(combinedimsview.(vec.(elem_fine)))
  Z = map(elem_to_dof, 1:N)
  L = zeros(T, (n+1), N*(p+1))
  for i=1:N
    L[X[:,:,i], Z[i]] .= L₁[Z₂, :]
  end
  H = (domain[2]-domain[1])/N
  L*(H/2)
end

"""
Function to compute the innerproduct of the LegendrePolynomials
"""
function assemble_legendre_mass_matrix(domain::SVector{2,T}, N::Int, p::Int) where T<:Real  
  t = [2/(2*T(j)-1) for j=1:p+1]
  H = (domain[2]-domain[1])/N
  Diagonal(repeat(t, outer=N))*(H/2)
end

end