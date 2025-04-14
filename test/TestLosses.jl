using Test
using KernelAbstractions
include("../src/losses/Loss.jl")
using Statistics
using Random
using PyCall
using CUDA

# Import MONAI modules
monai = pyimport("monai.losses")
torch = pyimport("torch")
np = pyimport("numpy")

# Set random seeds
Random.seed!(42)
np.random.seed(42)
torch.manual_seed(42)

@testset "MONAI Comparison Tests" begin
    # Test configurations for both 2D and 3D
    test_configs = [
        # (dims, channels, spatial_sizes, batch_size, name)
        (2, 1, (32, 32), 2, "2D"),        # 2D: (C,H,W,N) = (1,32,32,2)
        (3, 1, (16, 16, 16), 2, "3D")     # 3D: (C,D,H,W,N) = (1,16,16,16,2)
    ]
    
    for (n_dims, n_channels, spatial_sizes, batch_size, name) in test_configs
        @testset "Loss Functions - $name" begin
            # Generate data
            if n_dims == 2
                # 2D: (C,H,W,N)
                target = rand(Float32, n_channels, spatial_sizes..., batch_size)
                input = randn(Float32, n_channels, spatial_sizes..., batch_size)
                
                # Convert to MONAI format (N,C,H,W)
                target_monai = permutedims(target, (4, 1, 2, 3))
                input_monai = permutedims(input, (4, 1, 2, 3))
            else
                # 3D: (C,D,H,W,N)
                target = rand(Float32, n_channels, spatial_sizes..., batch_size)
                input = randn(Float32, n_channels, spatial_sizes..., batch_size)
                
                # Convert to MONAI format (N,C,D,H,W)
                target_monai = permutedims(target, (5, 1, 2, 3, 4))
                input_monai = permutedims(input, (5, 1, 2, 3, 4))
            end
            
            # Convert to PyTorch tensors
            target_torch = torch.tensor(target_monai, dtype=torch.float32)
            input_torch = torch.tensor(input_monai, dtype=torch.float32)
            
            @testset "Dice Loss" begin
                julia_loss = dice_loss(input, target, epsilon=1e-5, sigmoid=true)
                monai_loss = monai.DiceLoss(
                    sigmoid=true,
                    include_background=true
                )(input_torch, target_torch).item()
                
                println("$name Dice Test:")
                println("  Julia data shape: $(size(target))")
                println("  MONAI data shape: $(size(target_monai))")
                println("  Loss values: Julia=$julia_loss, MONAI=$monai_loss")
                
                @test isapprox(julia_loss, monai_loss, rtol=1e-3, atol=1e-3)
            end
            
            @testset "Jaccard Loss" begin
                julia_loss = jaccard_loss(input, target, epsilon=1e-5, sigmoid=true)
                monai_loss = monai.DiceLoss(
                    jaccard=true,
                    sigmoid=true,
                    include_background=true
                )(input_torch, target_torch).item()
                
                println("$name Jaccard Test:")
                println("  Julia data shape: $(size(target))")
                println("  MONAI data shape: $(size(target_monai))")
                println("  Loss values: Julia=$julia_loss, MONAI=$monai_loss")
                
                @test isapprox(julia_loss, monai_loss, rtol=1e-3, atol=1e-3)
            end
            
            @testset "Cross Entropy Loss" begin
                julia_loss = cross_entropy_loss(input, target, epsilon=1e-5, sigmoid=true)
                monai_loss = monai.DiceCELoss(
                    sigmoid=true,
                    include_background=true,
                    lambda_dice=0.0,
                    lambda_ce=1.0
                )(input_torch, target_torch).item()
                
                println("$name Cross Entropy Test:")
                println("  Julia data shape: $(size(target))")
                println("  MONAI data shape: $(size(target_monai))")
                println("  Loss values: Julia=$julia_loss, MONAI=$monai_loss")
                
                @test isapprox(julia_loss, monai_loss, rtol=1e-3, atol=1e-3)
            end
        end
    end
end