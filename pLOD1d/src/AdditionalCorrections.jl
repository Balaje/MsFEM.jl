module AdditionalCorrections

using Gridap
using Gridap.Geometry: get_cell_node_ids

using ProgressMeter
using StaticArrays

using pLOD1d.Triangulations: generate_triangulations, elements_in_coarse_scale_patch, get_interior
using pLOD1d.LegendrePolynomials: assemble_rectangular_matrix
using pLOD1d.MultiscaleBasis: multiscale_basis, multiscale_lhs, multiscale_rhs
using pLOD1d.StabilizedMultiscaleBasis: stabilized_multiscale_bases

"""
Function to compute the additional correction bases for time dependent problems:
  (Kalyanaraman, B., Krumbiegel, F., Maier, R., & Wang, S. (2025), arXiv [Math.NA])

Appends j-levels of corrections to the given multiscale basis β.
"""
function additional_correction_bases(β::Vector{U}, ntimes::Int, aₕ::Function, V::FESpace, domain::SVector{2, T}, n::Int, N::Int, l::Int, p::Int) where {T<:Real, U<:AbstractMatrix{<:Real}}  
  model_fine, model_coarse = generate_triangulations(domain, n, N);

  # Patch Connectivity info
  patch_coarse = elements_in_coarse_scale_patch(1:N, N, l);
  patch_fine = elements_in_coarse_scale_patch(get_cell_node_ids(model_fine), N, l);  
  
  # Needed to define the Mass matrix for L² inner product on the RHS.
  Ω = get_triangulation(V);
  dΩ = Measure(Ω, 2)

  K = assemble_matrix(aₕ, V, V);
  M = assemble_matrix((u,v)->∫(u*v)dΩ, V, V);
  L = assemble_rectangular_matrix(domain, n, N, p)  

  elem_to_dof(i) = (p+1)*(i-1)+1:(p+1)*i

  ms_basis = Vector{Vector{U}}(undef, ntimes+1);
  ms_basis[1] = β
  @showprogress for j=1:ntimes
    βⱼ = ms_basis[j]
    solⱼ = Vector{U}(undef, num_cells(model_coarse));    
    @showprogress "Computing additional corrections bases" for i=1:num_cells(model_coarse)
      # Same LHS as the multiscale basis      
      lhs, I_p = multiscale_lhs(K, L, patch_coarse[i], patch_fine[i], p)
      J_p = reduce(vcat, map(elem_to_dof, vec(patch_coarse[i]))) # Coarse Scale patch dofs
      
      # New RHS for the additional corrections
      rhs = [(M*βⱼ[i])[I_p, :]; zeros(T, length(J_p), p+1)];

      sol = zeros(T, n+1, p+1)
      sol[I_p, :] = (lhs\rhs)[1:length(I_p), :]
      solⱼ[i] = sol
    end
    ms_basis[j+1] = solⱼ
  end
  ms_basis
end;

"""
Function to obtain the additional corrections to the canonical basis
"""
function additional_correction_bases(ntimes::Int, aₕ::Function, V::FESpace, domain::SVector{2, T}, n::Int, N::Int, l::Int, p::Int) where T<:Real
  β = multiscale_bases(aₕ, V, domain, n, N, l, p)
  additional_correction_bases(β, ntimes, aₕ, V, domain, n, N, l, p)
end

"""
Function to obtain the additional corrections for the stabilized basis
"""
function stabilized_additional_correction_bases(ntimes::Int, aₕ::Function, V::FESpace, domain::SVector{2, T}, n::Int, N::Int, l::Int, p::Int) where T<:Real
  β = stabilized_multiscale_bases(aₕ, V, domain, n, N, l, p)
  additional_correction_bases(β, ntimes, aₕ, V, domain, n, N, l, p)
end

end