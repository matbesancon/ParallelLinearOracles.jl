

function FrankWolfe.compute_extreme_point(
    lmo::AcceleratedLinearOracle{<:FrankWolfe.LpNormBallLMO{T,Inf}}, direction; v=similar(direction, T),
) where {T}
    AK.foreachindex(direction) do idx
        v[idx] = -lmo.lmo.right_hand_side * (1 - 2signbit(direction[idx]))
    end
    return v
end
