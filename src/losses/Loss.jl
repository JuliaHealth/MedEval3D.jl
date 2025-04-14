using KernelAbstractions
using CUDA
using Statistics
using Tullio


"""
Convert between Julia and MONAI tensor formats
Julia:   (C, H, W, N) or (C, D, H, W, N)
MONAI:   (N, C, H, W) or (N, C, D, H, W)
"""
function permute_data_format(x)
    dims = ndims(x)
    if dims == 4  # 2D case
        return permutedims(x, (4, 1, 2, 3))  # (C,H,W,N) -> (N,C,H,W)
    elseif dims == 5  # 3D case
        return permutedims(x, (5, 1, 2, 3, 4))  # (C,D,H,W,N) -> (N,C,D,H,W)
    end
end


# Then modify the loss functions to handle the conversion:
function dice_loss(input, target; epsilon=1e-5, sigmoid=true)
    # Convert to MONAI format for comparison
    input_monai = permute_data_format(input)
    target_monai = permute_data_format(target)
    
    if sigmoid
        input_monai = 1.0f0 ./ (1.0f0 .+ exp.(-input_monai))
    end
    
    # Now batch dimension is first, making it easier to process
    batch_size = size(input_monai, 1)
    n_channels = size(input_monai, 2)
    
    total_loss = 0.0f0
    
    for b in 1:batch_size, c in 1:n_channels
        input_slice = selectdim(input_monai, 1, b) |> x -> selectdim(x, 1, c)
        target_slice = selectdim(target_monai, 1, b) |> x -> selectdim(x, 1, c)
        
        # Simple reduction operations without atomic operations
        intersection = sum(input_slice .* target_slice)
        sum_input = sum(input_slice)
        sum_target = sum(target_slice)
        
        dice = (2.0f0 * intersection + epsilon) / (sum_input + sum_target + epsilon)
        total_loss += (1.0f0 - dice)
    end
    
    return total_loss / (batch_size * n_channels)
end

"""
    jaccard_index(y_true::AbstractArray{<:Integer,N}, y_pred::AbstractArray{<:Integer,N}) where N

Calculates the Jaccard Index (Intersection over Union) based on integer masks.
Formula: intersection / union
"""
function jaccard_index(y_true::AbstractArray{T,N}, y_pred::AbstractArray{T,N}) where {T<:Integer, N}
    intersection = sum(y_true .& y_pred)
    union_val = sum(y_true .| y_pred)
    
    if union_val == 0
        return 1.0
    end
    
    return intersection / union_val
end

"""
    jaccard_loss(input, target; epsilon=1e-5, sigmoid=true)
"""
function jaccard_loss(input, target; epsilon=1e-5, sigmoid=true)
    # Convert to MONAI format
    input_monai = permute_data_format(input)
    target_monai = permute_data_format(target)
    
    if sigmoid
        input_monai = 1.0f0 ./ (1.0f0 .+ exp.(-input_monai))
    end
    
    batch_size = size(input_monai, 1)
    n_channels = size(input_monai, 2)
    
    total_loss = 0.0f0
    
    for b in 1:batch_size, c in 1:n_channels
        input_slice = selectdim(input_monai, 1, b) |> x -> selectdim(x, 1, c)
        target_slice = selectdim(target_monai, 1, b) |> x -> selectdim(x, 1, c)
        
        intersection = sum(input_slice .* target_slice)
        union_sum = sum(input_slice .+ target_slice .- input_slice .* target_slice)
        
        jaccard = (intersection + epsilon) / (union_sum + epsilon)
        total_loss += (1.0f0 - jaccard)
    end
    
    return total_loss / (batch_size * n_channels)
end

"""
    cross_entropy_loss(input, target; epsilon=1e-5, sigmoid=true)
"""
function cross_entropy_loss(input, target; epsilon=1e-5, sigmoid=true)
    # Convert to MONAI format
    input_monai = permute_data_format(input)
    target_monai = permute_data_format(target)
    
    if sigmoid
        input_monai = 1.0f0 ./ (1.0f0 .+ exp.(-input_monai))
    end
    
    batch_size = size(input_monai, 1)
    n_channels = size(input_monai, 2)
    
    total_loss = 0.0f0
    
    for b in 1:batch_size, c in 1:n_channels
        input_slice = selectdim(input_monai, 1, b) |> x -> selectdim(x, 1, c)
        target_slice = selectdim(target_monai, 1, b) |> x -> selectdim(x, 1, c)
        
        input_slice = clamp.(input_slice, epsilon, 1.0f0 - epsilon)
        ce = -target_slice .* log.(input_slice) .- (1.0f0 .- target_slice) .* log.(1.0f0 .- input_slice)
        total_loss += mean(ce)
    end
    
    return total_loss / (batch_size * n_channels)
end