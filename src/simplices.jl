
"""
    FrankWolfe.compute_extreme_point(lmo::AcceleratedLinearOracle{<:Union{FrankWolfe.HyperSimplexLMO, FrankWolfe.UnitHyperSimplexLMO}})

Accelerated version of the compute_extreme_point function for K-hypersimplices, using specialized sorting algorithms.
The `AcceleratedLinearOracle` `buffer` should be an `AbstractArray` of `Int` used to store the permutation indices.
"""
function FrankWolfe.compute_extreme_point(lmo::AcceleratedLinearOracle{HLMO}, direction; v=similar(direction)) where {HLMO <: Union{FrankWolfe.HyperSimplexLMO, FrankWolfe.UnitHyperSimplexLMO}}
    AK.sortperm!(lmo.buffer, direction; lmo.accelerated_options...)
    v .= 0
    K = _compute_k_hypersimplex(lmo.lmo, direction)
    v[lmo.buffer[1:K]] .= lmo.lmo.radius
    return v
end

_compute_k_hypersimplex(lmo::FrankWolfe.HyperSimplexLMO, direction) = lmo.K
_compute_k_hypersimplex(lmo::FrankWolfe.UnitHyperSimplexLMO, direction) = min(lmo.K, length(direction), sum(<(0), direction))

function FrankWolfe.compute_extreme_point(lmo::AcceleratedLinearOracle{SLMO}, direction; v=nothing) where {SLMO <: Union{FrankWolfe.ProbabilitySimplexLMO, FrankWolfe.UnitSimplexLMO}}
    # TODO replace by mapreduce implementation at some point
    (val, idx_min) = findmin(direction)
    val = _compute_simplex_val(lmo.lmo, val)
    return FrankWolfe.ScaledHotVector(val, idx_min, length(direction))
end

_compute_simplex_val(lmo::FrankWolfe.ProbabilitySimplexLMO, val) = lmo.right_side
_compute_simplex_val(lmo::FrankWolfe.UnitSimplexLMO, val) = ifelse(val <= 0, lmo.right_side, zero(lmo.right_side))
