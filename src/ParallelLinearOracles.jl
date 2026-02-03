module ParallelLinearOracles

import AcceleratedKernels as AK
import FrankWolfe

struct AcceleratedLinearOracle{LMO,KwargsType,BT} <: FrankWolfe.LinearMinimizationOracle
    lmo::LMO
    accelerated_options::KwargsType
    buffer::BT
end

include("simplices.jl")
include("norm_balls.jl")

end
