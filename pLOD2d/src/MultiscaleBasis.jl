module MultiscaleBasis

using ProgressMeter
using SparseArrays
using Gridap
using StaticArrays
using SplitApplyCombine

using Gridap.Geometry: get_cell_node_ids

using pLOD2d.Triangulations: generate_triangulations, elements_in_coarse_scale_patch, get_interior
using pLOD2d.LegendrePolynomials: assemble_legendre_mass_matrix, assemble_rectangular_matrix

"""
Function to generate the multiscale basis functions:
   (Maier, R. 2021, SIAM Journal on Numerical Analysis 59(2), 1067-1089)
"""
function multiscale_basis(aₕ::Function, V::FESpace, domain::SVector{N1, T}, n::Int, N::Int, l::Int, p::Int) where {N1, T<:Real}  
  model_fine, model_coarse = generate_triangulations(domain, n, N);
  patch_coarse = elements_in_coarse_scale_patch(reshape(1:N*N, N, N), N, l);
  patch_fine = elements_in_coarse_scale_patch(get_cell_node_ids(model_fine), N, l);

  # RHS is the inner product of the Legendre Polynomials.
  O = assemble_legendre_mass_matrix(N, p, T(0))
  K = assemble_matrix(aₕ, V, V);
  L = assemble_rectangular_matrix(domain, n, N, p)

  ms_basis = Matrix{T}[];
  @showprogress "Computing pLOD basis" for i=1:num_cells(model_coarse)
    lhs, free_dofs = multiscale_lhs(K, L, patch_coarse[i], patch_fine[i], p)
    rhs = [zeros(T, length(free_dofs), (p+1)*(p+1)); 
           multiscale_rhs(O, patch_coarse[i], p, i)];           
    sol = zeros(T, (n+1)*(n+1), (p+1)*(p+1))
    sol[free_dofs, :] = (lhs\rhs)[1:length(free_dofs), :]
    push!(ms_basis, sol)
  end
  ms_basis
end;

"""
Function to generate the LHS of the saddle point problem to compute the canonical multiscale bases.
"""
function multiscale_lhs(K::AbstractMatrix{T1}, L::AbstractMatrix{T1}, patch_coarse::AbstractVecOrMat{T2}, patch_fine::AbstractVecOrMat{T3}, p::Int) where {T1<:Real, T2<:Integer, T3<:Vector{<:Integer}}

  elem_to_dof(i) = (p+1)^2*(i-1)+1:(p+1)^2*i
  
  # Patch degrees of freedom
  I_p = unique(combinedims(get_interior(patch_fine))); # Fine Scale patch dofs
  J_p = reduce(vcat, map(elem_to_dof, vec(patch_coarse))) # Coarse Scale patch dofs

  # Zero Matrix
  O_p = zeros(T1, length(J_p), length(J_p))

  # Final matrix
  LHS = [K[I_p, I_p] L[I_p, J_p]; 
         L[I_p, J_p]' O_p]

  LHS, I_p
end;

"""
Function to generate the RHS of the saddle point problem to compute the canonical multiscale bases.
"""
function multiscale_rhs(O::AbstractMatrix{T1}, patch_coarse::AbstractVecOrMat{T2}, p::Int, i::Int) where {T1, T2<:Integer}
  elem_to_dof(i) = (p+1)^2*(i-1)+1:(p+1)^2*i
  J_p = reduce(vcat, map(elem_to_dof, vec(patch_coarse)))
  I_p = elem_to_dof(i)
  O[J_p, I_p]
end;

end