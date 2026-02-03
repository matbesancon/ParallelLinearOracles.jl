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

@testset "Norm balls" begin
    n = 10
    direction = randn(n)
    radius = 3.0
    @testset "Norm Inf $T" for T in (Int, Float32)
        lmo = FrankWolfe.LpNormBallLMO{T, Inf}(radius)
        lmo_acc = ParallelLinearOracles.AcceleratedLinearOracle(
            lmo,
            (; max_tasks=4),
            nothing,
        )
        v = FrankWolfe.compute_extreme_point(lmo, direction)
        v_acc = FrankWolfe.compute_extreme_point(lmo_acc, direction)
        @test v ≈ v_acc
        @test eltype(v_acc) == T
    end
end

@testset "CUDA" begin
    if CUDA.functional()
        n = 10
        direction = CUDA.randn(n)
        K = 4
        radius = 3.5f0
        @testset "$base_lmo" for base_lmo in (
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

        @testset "$base_lmo" for base_lmo in (
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
        @testset "Norm Inf $T" for T in (Float16, Float32)
            lmo = FrankWolfe.LpNormBallLMO{T, Inf}(radius)
            lmo_acc = ParallelLinearOracles.AcceleratedLinearOracle(
                lmo,
                (;),
                nothing,
            )
            v = FrankWolfe.compute_extreme_point(lmo, collect(direction))
            v_acc = FrankWolfe.compute_extreme_point(lmo_acc, direction)
            @test v ≈ collect(v_acc)
            @test eltype(v_acc) == T
        end
    end
end
