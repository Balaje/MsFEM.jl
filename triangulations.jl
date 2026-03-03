using StaticArrays, BlockArrays, SplitApplyCombine

using Gridap
using Gridap.Geometry: get_cell_node_ids

"""
Function to generate a pair of triangulations on the coarse and fine scale.
"""
function generate_triangulations(domain::SVector{N1,T1}, n::T2, N::T3) where {N1, T1<:Real, T2<:Int, T3<:Int}
  CartesianDiscreteModel(domain, (n,n)), CartesianDiscreteModel(domain, (N,N))
end;

function elements_in_coarse_scale_patch(cells::AbstractMatrix{T}, N::Int, l::Int) where T
  n = size(cells, 1) 
  a = ceil(Int, n/N)  
  cells_coarse = BlockedArray(cells, fill(a, N), fill(a,N))
  cells_l_patch = []
  for J=1:N, I=1:N    
    sI = Block(max(I-l, 1))
    eI = Block(min(I+l, N))
    sJ = Block(max(J-l, 1))
    eJ = Block(min(J+l, N))
    push!(cells_l_patch, collect(cells_coarse[sI:eI, sJ:eJ]))
  end
  cells_l_patch
end

"""
Function to get the interior elements
"""
function get_interior(a::AbstractMatrix{T}) where T
  vec(a[2:end-1, 2:end-1])
end;