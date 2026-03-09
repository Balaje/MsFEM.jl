module StabilizedMultiscaleBasis

using Gridap
using Gridap.Arrays: Fill
using Gridap.Geometry: get_cell_node_ids

using StaticArrays
using SplitApplyCombine
using BlockArrays
using SparseArrays

using ProgressMeter

using pLOD1d.Triangulations: elements_in_coarse_scale_patch, get_interior
using pLOD1d.LegendrePolynomials: assemble_rectangular_matrix
using pLOD1d.MultiscaleBasis: multiscale_bases, multiscale_lhs, multiscale_rhs

"""
Function to compute the stabilization of the multiscale bases:
  (Dong, Z., Hauck, M., & Maier, R. (2023), SIAM Journal on Numerical Analysis, 61(4), 1918–1937)

Modifies the basis corresponding to the constant Legendre polynomial Λ₀,ₖ
"""
function stabilized_multiscale_bases(aₕ::Function, V::FESpace, domain::SVector{2, T}, n::Int, N::Int, l::Int, p::Int) where T<:Real
  
  model_fine = CartesianDiscreteModel(domain, (n,))

  # Element-level Coarse to Fine Map.
  elem_fine_nodes = elements_in_coarse_scale_patch(get_cell_node_ids(model_fine), N, 0);
  elem_fine = elements_in_coarse_scale_patch(1:n, N, 0);

  # Connectivity for Patch of Radius 1.
  patch_1 = elements_in_coarse_scale_patch(1:N, N, 1);  
  patch_1_fine = elements_in_coarse_scale_patch(get_cell_node_ids(model_fine), N, 1);

  # Connectivity for Patch of Radius l.
  patch_l_coarse = elements_in_coarse_scale_patch(1:N, N, l);
  patch_l_fine = elements_in_coarse_scale_patch(get_cell_node_ids(model_fine), N, l);  
  
  # Stiffness and the rectangular matrix
  stima = assemble_matrix(aₕ, V, V);
  lmat = assemble_rectangular_matrix(domain, n, N, p);

  # Mesh size
  H = (domain[2]-domain[1])/N;

  # Hat functions on the coarse scale reference domain
  ref_domain = @SVector T[-1,1]
  ϕₘ = ϕ(ref_domain, n, N);

  # Multiscale Bases
  β = multiscale_bases(aₕ, V, domain, n, N, l, p);    
  α = Vector{Vector{T}}(undef, N)
  
  elem_to_dof(i) = (p+1)*(i-1)+1:(p+1)*i
  
  @showprogress "Computing stabilized-pLOD bases" for K = 1:N
    sol = zeros(T, n+1)   
    
    patch_1_cells = split_patch(patch_1[K])   

    ## Compute ιₖ
    iota_ref = ι(ref_domain, n, N, size(patch_1_cells))
    iota = assemble_patch_vector(iota_ref, patch_1_fine[K], n+1)    

    ## Compute Cˡιₖ
    for j=1:lastindex(patch_1_cells)
      patch_1_el = patch_1_cells[j]               
      for i=1:lastindex(patch_1_el)
        el = patch_1_el[i]
        free_dofs_coarse = reduce(vcat, map(elem_to_dof, vec(patch_l_coarse[el])))        
        lhs, free_dofs_fine = multiscale_lhs(stima, lmat, patch_l_coarse[el], patch_l_fine[el], p)         
        K_el = assemble_patch_matrix(aₕ, V, elem_fine_nodes[el], elem_fine[el])              
        rhs₁ = assemble_patch_vector(ϕₘ[i], elem_fine_nodes[el], n+1)
        rhs₂ = zeros(T, length(free_dofs_coarse));
        rhs = [(K_el*rhs₁)[free_dofs_fine]; rhs₂];        
        sol[free_dofs_fine] += (lhs\rhs)[1:length(free_dofs_fine)];
      end
    end

    # Compute (1-Cˡ)ιₖ
    sol = iota - sol;    

    # Compute (1-Cˡ)νₖ    
    δ = 1/(length(patch_1[K]));    
    C = 2/H*lmat'*iota
    Z = zero(C)
    Z[(p+1)*(K-1)+1] = 2; # ( = 2/H*λ(N,p)[1,1] );
    D = reshape((Z - C)*δ, p+1, N);            
    sol1 = zeros(T, n+1)
    for i=1:lastindex(patch_1[K])  
      el = patch_1[K][i]
      sol1 += β[el]*D[:,el]
    end    

    # (1-Cˡ)(ιₖ+νₖ)    
    α[K] = sol + sol1;
  end
  
  # Assign the 0-th order basis to the new ones
  for i=1:N
    β[i][:,1] = α[i]
  end

  β
end;

"""
Function to assemble the vector (rhs) on the cell nodes (cell_nodes) and return a vector on the fine scale discretization (ndofs)
"""
function assemble_patch_vector(rhs::AbstractVecOrMat{T}, cell_nodes::AbstractVector{U}, ndofs::Int) where {T<:Vector{<:Real}, U<:Vector{<:Integer}}
  m = length(cell_nodes)
  nz_vals = 2*m
  Is = Vector{Int}(undef, nz_vals);
  Vs = T(undef, nz_vals);
  ct = 1
  for i=1:lastindex(cell_nodes)
    C = cell_nodes[i]
    rhs_el = rhs[i]
    for j=1:2
      Is[ct] = C[j]
      Vs[ct] = rhs_el[j]
      ct += 1
    end
  end  
  d = Dict(Is .=> Vs)
  collect(sparsevec(d, ndofs))  
end

"""
Function to assemble the matrix corresponding to the bilinear form (aₕ) on the cell nodes (cell_node_ids). 
Also requires the fine scale cells (cell) and the background fine scale space (V).
"""
function assemble_patch_matrix(aₕ::Function, V::FESpace, cell_node_ids::AbstractVector{T}, cells::AbstractVector{U}) where {T<:Vector{<:Integer}, U<:Integer}
  b₀ = get_fe_basis(V);
  b = get_trial_fe_basis(V);
  Ω = get_triangulation(V);
  Ks = aₕ(b, b₀)[Ω];
  W = eltype(first(Ks))
  m = length(cell_node_ids);
  nz_vals = 2*2*m;
  Is = T(undef, nz_vals);
  Js = T(undef, nz_vals);
  Vs = Vector{W}(undef, nz_vals);
  ct = 1;
  for i=1:m
    C = cell_node_ids[i]
    Kel = Ks[cells[i]]
    for j=1:2, k=1:2
      Is[ct] = C[j]
      Js[ct] = C[k]
      Vs[ct] = Kel[j,k]
      ct += 1
    end
  end
  ndofs = num_free_dofs(V);  
  sparse(Is, Js, Vs, ndofs, ndofs)  
end

"""
Nodal basis functions on the reference square (-1,1)²
"""
function ϕ(x::VectorValue{1, T}) where T<:Real
  ξ, = x
  @SVector[0.5*(1+ξ), 0.5*(1-ξ)];
end;

"""
Reference coarse-scale nodal basis on the fine-scale nodes contained inside the coarse elements.
"""
function ϕ(domain::SVector{2,T}, n::Int, N::Int) where T<:Real
  a = ceil(Int, n/N)
  model = CartesianDiscreteModel(domain, (a,))
  cell_coords = get_cell_coordinates(model)
  ϕᵢ = map(Broadcasting(ϕ), cell_coords)
  @inline get_ith_basis(X, i) = map(getindex, X, Fill(i, size(X)))
  basis_vals_on_cell = [0.5*get_ith_basis.(ϕᵢ, i) for i=1:2]    
  basis_vals_on_cell
end

"""
Function to compute the reference ι(x) function as a sum of nodal basis functions corresponding to the interior nodes.
"""
function ι(X::AbstractVecOrMat{T}, patch_size::NTuple{1, Int}) where T
  if(patch_size == (1,))
    return [X[1]; X[2]]
  elseif(patch_size == (2,))
    return [X[1]; X[2] + X[1]; X[2]]
  end
end

"""
The reference 1-patch function ι(x) on the coarse scale. The function is evaluated on the fine scale.
"""
function ι(domain::SVector{2,T}, n::Int, N::Int, patch_size::NTuple{1, Int}) where T<:Real
  X = ϕ(domain, n, N)
  ι(X, patch_size)
end

"""
Functions to split the patch into blocks of 4 elements: For example:

[1 2 3; 4 5 6; 7 8 9] =  [ [[1 2; 4 5]]  [[2 3; 5 6]]; [[4 5; 7 8]]  [[5 6; 8 9]] ]

1. First duplicate the even rows inside the patch numering, (\\_duplicate\\_even\\_row\\_col)
2. Use BlockedArrays.jl to convert the matrix into vector of matrices. (_blockify)
"""
function split_patch(A::Vector{T}) where T
  (_blockify∘_duplicate_even_row_col)(A)
end

function _duplicate_even_row_col(A::Vector{T}) where T
  blocks = T[]  
  n = length(A)
  for i=1:n
    ((i!=n)||(i!=1)) && push!(blocks, A[i])
    if(iseven(i) && (i!=n))
      push!(blocks, A[i])
    end
  end  
  blocks
end;

function _blockify(A::Vector{T}) where T
  M = length(A)
  m = ceil(Int, M/2)  
  blocks = fill(2, m)  
  B = BlockArray(A, blocks)  
  C = Vector{T}[]  
  for i=1:m
    push!(C, B[Block(i)])
  end
  C
end;

end