module Triangulations

using StaticArrays, BlockArrays, SplitApplyCombine

using Gridap
using Gridap.Geometry: get_cell_node_ids

"""
Function to generate a pair of Gridap triangulations on the coarse and fine scale.
"""
function generate_triangulations(domain::SVector{N1,T1}, n::T2, N::T3) where {N1, T1<:Real, T2<:Int, T3<:Int}
  CartesianDiscreteModel(domain, (n,)), CartesianDiscreteModel(domain, (N,))
end;

"""
Function that accepts the cell-wise entries (node or elements) and returns the corresponding entries on the l-patch of the coarse scale 
"""
function elements_in_coarse_scale_patch(cells::AbstractVector{T}, N::Int, l::Int) where T
  n = size(cells, 1) 
  a = ceil(Int, n/N)  
  cells_coarse = BlockedArray(cells, fill(a, N))
  cells_l_patch = AbstractVector{T}[]
  for I=1:N    
    sI = Block(max(I-l, 1))
    eI = Block(min(I+l, N))        
    push!(cells_l_patch, collect(cells_coarse[sI:eI]))
  end
  cells_l_patch
end;

"""
Function to get the interior elements
"""
function get_interior(a::AbstractVector{T}) where T
  vec(a[2:end-1])
end;

end