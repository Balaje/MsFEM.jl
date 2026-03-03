"""
Function to compute the additional correction bases.
"""
function additional_correction_bases(β::Vector{U}, ntimes::Int, aₕ::Function, V::FESpace, domain::SVector{N1, T}, n::Int, N::Int, l::Int, p::Int) where {N1, T, U}  
  model_fine, model_coarse = generate_triangulations(domain, n, N);
  patch_coarse = elements_in_coarse_scale_patch(reshape(1:N*N, N, N), N, l);
  patch_fine = elements_in_coarse_scale_patch(get_cell_node_ids(model_fine), N, l);  
  K = assemble_matrix(aₕ, V, V);
  Ω = get_triangulation(V);
  dΩ = Measure(Ω, 2)
  M = assemble_matrix((u,v)->∫(u*v)dΩ, V, V);
  L = assemble_L_matrix(model_fine, n, N, p)  

  elem_to_dof(i) = (p+1)^2*(i-1)+1:(p+1)^2*i

  ms_basis = [β];
  @showprogress for j=1:ntimes
    βⱼ = ms_basis[j]
    solⱼ = [];    
    @showprogress for i=1:num_cells(model_coarse)
      lhs, I_p = multiscale_lhs(K, L, patch_coarse[i], patch_fine[i], p)
      J_p = reduce(vcat, map(elem_to_dof, vec(patch_coarse[i]))) # Coarse Scale patch dofs
      
      rhs = [(M*βⱼ[i])[I_p, :]; zeros(length(J_p), (p+1)*(p+1))];

      sol = zeros(T, (n+1)*(n+1), (p+1)*(p+1))
      sol[I_p, :] = (lhs\rhs)[1:length(I_p), :]
      push!(solⱼ, sol)
    end
    push!(ms_basis, solⱼ)
  end
  ms_basis
end;