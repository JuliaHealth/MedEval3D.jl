using Test
using Random
using CUDA
using KernelAbstractions
using Statistics
using PyCall

# Import MONAI for comparison
const monai = PyCall.pyimport("monai.losses")
const torch = PyCall.pyimport("torch")

# Set random seed for reproducibility
Random.seed!(42)

@testset "Loss Functions - GPU" begin
    # Test configurations for both 2D and 3D
    test_configs = [
        # (dims, channels, spatial_sizes, batch_size, name)
        (2, 1, (32, 32), 2, "2D"),        # 2D: (N,C,H,W) = (2,1,32,32)
        (3, 1, (16, 16, 16), 2, "3D")     # 3D: (N,C,D,H,W) = (2,1,16,16,16)
    ]
    
    for (n_dims, n_channels, spatial_sizes, batch_size, name) in test_configs
        @testset "Loss Functions - $name" begin
            # Create test data directly on GPU
            if n_dims == 2
                input = CUDA.rand(Float32, batch_size, n_channels, spatial_sizes...)
                target = Float32.(CUDA.rand(batch_size, n_channels, spatial_sizes...) .> 0.5)
                
                # Create PyTorch tensors on GPU
                input_torch = torch.tensor(Array(input), requires_grad=true, device="cuda")
                target_torch = torch.tensor(Array(target), device="cuda")
            else
                input = CUDA.rand(Float32, batch_size, n_channels, spatial_sizes...)
                target = Float32.(CUDA.rand(batch_size, n_channels, spatial_sizes...) .> 0.5)
                
                # Create PyTorch tensors on GPU
                input_torch = torch.tensor(Array(input), requires_grad=true, device="cuda")
                target_torch = torch.tensor(Array(target), device="cuda")
            end
            
            @testset "Dice Loss" begin
                # Create MONAI loss function on GPU
                monai_loss = monai.DiceLoss(
                    sigmoid=true,
                    include_background=true
                ).cuda()
                
                # Compute losses
                julia_loss = dice_loss(input, target, sigmoid=true)
                monai_loss_val = monai_loss(input_torch, target_torch).item()
                
                @test isapprox(julia_loss, monai_loss_val, rtol=1e-3)
            end
            @testset "Jaccard Loss" begin
              # Create MONAI loss function on GPU
                monai_loss = monai.DiceLoss(
                  sigmoid=true,
                  include_background=true,
                  jaccard=true
                ).cuda()
                
                # Compute losses
                julia_loss = jaccard_loss(input, target, sigmoid=true)
                monai_loss_val = monai_loss(input_torch, target_torch).item()

                @test isapprox(julia_loss, monai_loss_val, rtol=1e-3)
            end

            @testset "Cross Entropy Loss" begin
                # Create MONAI loss function on GPU
                monai_loss = monai.DiceCELoss(
                    sigmoid=true,
                    include_background=true,
                    lambda_dice=0.0,
                    lambda_ce=1.0
                ).cuda()
                
                # Compute losses
                julia_loss = cross_entropy_loss(input, target, sigmoid=true)
                monai_loss_val = monai_loss(input_torch, target_torch).item()
                
                @test isapprox(julia_loss, monai_loss_val, rtol=1e-3)
            end
            
            # Clean up GPU memory
            CUDA.reclaim()
        end
    end
end