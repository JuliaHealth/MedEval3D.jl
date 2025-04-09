using Test
using KernelAbstractions
include("../src/losses/Loss.jl")
using Statistics

@testset "Loss Functions (CPU)" begin

    @testset "Dice Loss" begin
        y_true = [1 1 0; 0 1 0; 0 0 0]
        y_pred = [1 0 0; 0 1 1; 0 0 0]

        # Calculation:
        # intersection = sum(y_true .& y_pred) = sum([1 0 0; 0 1 0; 0 0 0]) = 2
        # denom = sum(y_true) + sum(y_pred) = 3 + 3 = 6
        # dice_coeff = 2.0 * 2 / 6 = 4 / 6 = 2/3
        # dice_loss = 1.0 - 2/3 = 1/3
        expected_loss = 1.0 / 3.0
        @test dice_loss(y_true, y_pred) ≈ expected_loss 

        # Test case: Empty masks
        y_true_empty = [0 0; 0 0]
        y_pred_empty = [0 0; 0 0]
        @test dice_loss(y_true_empty, y_pred_empty) ≈ 0.0

        # Test case: Perfect match
        y_true_perf = [1 1; 0 1]
        y_pred_perf = [1 1; 0 1]
        @test dice_loss(y_true_perf, y_pred_perf) ≈ 0.0

        # Test case: Complete mismatch
        y_true_mis = [1 1; 0 0]
        y_pred_mis = [0 0; 1 1]
        # intersection = 0, denom = 2 + 2 = 4, loss = 1.0 - 0 = 1.0
        @test dice_loss(y_true_mis, y_pred_mis) ≈ 1.0
    end

    @testset "Jaccard Loss" begin
        y_true = [1 1 0; 0 1 0; 0 0 0]
        y_pred = [1 0 0; 0 1 1; 0 0 0]

        # Calculation:
        # intersection = sum(y_true .& y_pred) = sum([1 0 0; 0 1 0; 0 0 0]) = 2
        # union_val = sum(y_true .| y_pred) = sum([1 1 0; 0 1 1; 0 0 0]) = 4
        # jaccard_index = 2 / 4 = 0.5
        # jaccard_loss = 1.0 - 0.5 = 0.5
        expected_loss = 0.5
        @test jaccard_loss(y_true, y_pred) ≈ expected_loss 

        # Test case: Empty masks
        y_true_empty = [0 0; 0 0]
        y_pred_empty = [0 0; 0 0]
        @test jaccard_loss(y_true_empty, y_pred_empty) ≈ 0.0 # 1 - jaccard_index(1.0)

        # Test case: Perfect match
        y_true_perf = [1 1; 0 1]
        y_pred_perf = [1 1; 0 1]
        @test jaccard_loss(y_true_perf, y_pred_perf) ≈ 0.0 # 1 - jaccard_index(1.0)

        # Test case: Complete mismatch
        y_true_mis = [1 1; 0 0]
        y_pred_mis = [0 0; 1 1]
        # intersection = 0, union_val = sum([1 1; 1 1]) = 4
        # jaccard_index = 0 / 4 = 0.0
        # jaccard_loss = 1.0 - 0.0 = 1.0
        @test jaccard_loss(y_true_mis, y_pred_mis) ≈ 1.0
    end

    @testset "Cross Entropy Loss" begin
        y_true = [1.0 0.0; 1.0 1.0]
        y_pred = [0.9 0.1; 0.8 0.7]
        epsilon = 1e-12

        # Manual calculation:
        # -(1*log(0.9) + (1-1)*log(1-0.9)) = -log(0.9)
        # -(0*log(0.1) + (1-0)*log(1-0.1)) = -log(0.9)
        # -(1*log(0.8) + (1-1)*log(1-0.8)) = -log(0.8)
        # -(1*log(0.7) + (1-1)*log(1-0.7)) = -log(0.7)
        # loss = mean([-log(0.9), -log(0.9), -log(0.8), -log(0.7)])
        expected_loss = mean([-log(0.9), -log(0.9), -log(0.8), -log(0.7)])

        @test cross_entropy_loss(y_true, y_pred; epsilon=epsilon) ≈ expected_loss 

        # Test with integer true values
        y_true_int = [1 0; 1 1]
        @test cross_entropy_loss(y_true_int, y_pred; epsilon=epsilon) ≈ expected_loss

        # Test edge case: predictions close to 0 and 1
        y_pred_edge = [1.0-epsilon/2 epsilon/2; 1.0-epsilon/2 epsilon/2]
        y_true_edge = [1 0; 1 0]

        loss_edge = cross_entropy_loss(y_true_edge, y_pred_edge; epsilon=epsilon)
        @test isfinite(loss_edge)
        # Manual calculation with clamping:
        # y_pred_clamped = [1-eps eps; 1-eps eps]
        # loss_00 = -(1*log(1-eps) + 0*log(eps)) = -log(1-eps)
        # loss_01 = -(0*log(eps) + 1*log(1-eps)) = -log(1-eps)
        # loss_10 = -(1*log(1-eps) + 0*log(eps)) = -log(1-eps)
        # loss_11 = -(0*log(eps) + 1*log(1-eps)) = -log(1-eps)
        # mean_loss = -log(1-eps)
        @test loss_edge ≈ -log(1.0-epsilon)
    end

    # Hausdorff Distance Transform Loss Tests
    @testset "Hausdorff DT Loss" begin
        y_true = [1 0 0; 0 1 0; 0 0 0]

        y_pred = Float32[0.9 0.2 0.1; 0.1 0.8 0.3; 0.2 0.1 0.1] 
        
        # Expected Distance Transform for y_true
        # Our implementation gives:
        # - Points where y_true == 1 get distance 0
        # - Other points get distance = shortest path to a 1
        # For this 3x3 example with 1s at (1,1) and (2,2):
        # [0 1 2]
        # [1 0 1]
        # [2 1 2]
        expected_dt = Float32[ 
            0.0 1.0 2.0;
            1.0 0.0 1.0;
            2.0 1.0 2.0 
        ]
        
        y_true_float = Float32.(y_true)
        loss_components = ((y_pred - y_true_float).^2) .* expected_dt
        expected_hausdorff_loss = mean(loss_components)
        
        # Test the full loss calculation
        actual_loss = hausdorff_dt_loss(y_pred, y_true)
        @test actual_loss ≈ expected_hausdorff_loss rtol=1e-5
        
        # Test the distance transform separately
        backend = KernelAbstractions.CPU()
        actual_dt = distance_transform_ka(backend, y_true)
        @test all(actual_dt .≈ expected_dt)
        
        # Additional test: Perfect prediction should give zero loss
        y_pred_perfect = Float32.(y_true)
        perfect_loss = hausdorff_dt_loss(y_pred_perfect, y_true)
        @test perfect_loss ≈ 0.0 atol=1e-6
    end

end