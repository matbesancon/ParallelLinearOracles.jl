using ParallelLinearOracles
using Test
import AcceleratedKernels as AK
using FrankWolfe
using CUDA

@testset "Simplex oracles" begin
    n = 10
    direction = randn(n)
    K = 4
    radius = 3.5
    @testset for base_lmo in (
        FrankWolfe.HyperSimplexLMO(K, radius),
        FrankWolfe.UnitHyperSimplexLMO(K, radius),
    )
        almo = ParallelLinearOracles.AcceleratedLinearOracle(
            base_lmo,
            (; max_tasks=4),
            zeros(Int, n),
        )
        v_a = FrankWolfe.compute_extreme_point(almo, direction)
        v = FrankWolfe.compute_extreme_point(base_lmo, direction)
        @test v ≈ v_a
    end

    @testset for base_lmo in (
        FrankWolfe.ProbabilitySimplexLMO(radius),
        FrankWolfe.UnitSimplexLMO(radius),
    )
        almo = ParallelLinearOracles.AcceleratedLinearOracle(
            base_lmo,
            (; max_tasks=4),
            nothing,
        )
        v_a = FrankWolfe.compute_extreme_point(almo, direction)
        v = FrankWolfe.compute_extreme_point(base_lmo, direction)
        @test v ≈ v_a
    end
end

@testset "CUDA" begin
    if CUDA.functional()
        n = 10
        direction = CUDA.randn(n)
        K = 4
        radius = 3.5f0
        @testset for base_lmo in (
            FrankWolfe.HyperSimplexLMO(K, radius),
            FrankWolfe.UnitHyperSimplexLMO(K, radius),
        )
            almo = ParallelLinearOracles.AcceleratedLinearOracle(
                base_lmo,
                (; temp=CUDA.zeros(Int32, n)),
                CUDA.zeros(Int32, n),
            )
            v_a = FrankWolfe.compute_extreme_point(almo, direction)
            v = FrankWolfe.compute_extreme_point(base_lmo, collect(direction))
            @test v ≈ collect(v_a)
        end

        @testset for base_lmo in (
            FrankWolfe.ProbabilitySimplexLMO(radius),
            FrankWolfe.UnitSimplexLMO(radius),
        )
            almo = ParallelLinearOracles.AcceleratedLinearOracle(
                base_lmo,
                (; temp=CUDA.zeros(Int32, n)),
                nothing,
            )
            v_a = FrankWolfe.compute_extreme_point(almo, direction)
            v = FrankWolfe.compute_extreme_point(base_lmo, collect(direction))
            @test v ≈ collect(v_a)
        end
    end
end
