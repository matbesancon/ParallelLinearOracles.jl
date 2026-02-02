

function FrankWolfe.compute_extreme_point(
    lmo::AcceleratedLinearOracle{BallLMO}, direction; v=similar(direction),
) where {BallLMO <: Union{FrankWolfe.LpNormBall{<:Real,Inf}}}

    AK.foreachindex(direction) do idx
        v[idx] = -lmo.right_hand_side * (1 - 2signbit(direction[idx]))
    end
    return v
end
