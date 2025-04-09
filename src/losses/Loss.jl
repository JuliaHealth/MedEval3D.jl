using KernelAbstractions
using CUDA
using Statistics
using Tullio

const KA = KernelAbstractions


@kernel function init_dt_kernel!(dt, y, T)
    idx = @index(Global, Cartesian)
    dt[idx] = y[idx] > 0 ? zero(T) : typemax(T)/2 # Initialize with 0 or ~Inf
end

# Kernel for one pass of distance propagation (Manhattan/Chebyshev-like update)
@kernel function dt_pass_kernel!(dt_out, dt_in, dims)
    idx = @index(Global, Cartesian)
    
    current_val = dt_in[idx]
    min_neighbor_val = current_val
    
    # Check neighbors in all dimensions
    for dim in 1:length(dims)
        # Check neighbor +1
        neighbor_idx_plus = Base.setindex(idx.I, idx.I[dim] + 1, dim)
        if checkbounds(Bool, dt_in, neighbor_idx_plus...)
             min_neighbor_val = min(min_neighbor_val, dt_in[neighbor_idx_plus...] + 1)
        end
        # Check neighbor -1
        neighbor_idx_minus = Base.setindex(idx.I, idx.I[dim] - 1, dim)
        if checkbounds(Bool, dt_in, neighbor_idx_minus...)
             min_neighbor_val = min(min_neighbor_val, dt_in[neighbor_idx_minus...] + 1)
        end
    end
    
    dt_out[idx] = min_neighbor_val
end

"""
    distance_transform_ka(backend, y::AbstractArray{U,N}; iterations=size(y,1), T=Float32) where {U,N}

Simplified Distance Transform using KernelAbstractions.
Performs multiple passes propagating distance +1 from neighbors.
Approximates Manhattan (L1) or Chebyshev (Linf) distance, NOT Euclidean (L2).

Args:
    backend: KernelAbstractions backend.
    y: Input mask (integer or boolean AbstractArray).
    iterations: Number of propagation passes. Defaults to the size of the first dimension.
                More iterations give better approximation but cost more.
    T: Float type for the output distance transform array (default: Float32).

Returns:
    Distance transform array (KA-compatible, e.g., CuArray) of type T.
"""
function distance_transform_ka(backend, y::AbstractArray{U,N}; iterations=size(y,1), T=Float32) where {U<:Union{Integer,Bool}, N}
    if N > 3
        @warn "Simplified DT kernel currently only checks immediate neighbors (up to 3D). Results might be less accurate for >3D."
    end
    
    dims = size(y)
    # Allocate initial distance transform array
    dt_current = KA.zeros(backend, T, dims)
    
    # Initialize distances: 0 for foreground (y > 0), ~Inf for background
    init_kernel! = init_dt_kernel!(backend)
    init_kernel!(dt_current, y, T, ndrange=dims)
    KernelAbstractions.synchronize(backend)
    
    # Allocate second buffer for ping-ponging passes
    dt_next = KA.similar(dt_current)
    
    # Perform multiple propagation passes
    # This approximates the distance by letting the '0's expand outwards
    pass_kernel! = dt_pass_kernel!(backend)
    for _ in 1:iterations
        pass_kernel!(dt_next, dt_current, dims, ndrange=dims)
        KernelAbstractions.synchronize(backend)
        # Swap buffers for next iteration
        dt_current, dt_next = dt_next, dt_current 
    end
    
    return dt_current
end

@kernel function hausdorff_dt_loss_kernel!(loss_components, y_pred, y, dt, epsilon)
    idx = @index(Global, Linear)
    
    # MONAI formula: ((y_pred - y)**2 * dt)
    
    # Ensure y is Float
    y_float = convert(eltype(y_pred), y[idx])
    
    # multiplied by distance : pixels far from the boundary get weighted more if they are incorrectly predicted.
    loss_components[idx] = (y_pred[idx] - y_float)^2 * dt[idx]
end

"""
    hausdorff_dt_loss(y_pred::AbstractArray{T,N}, y::AbstractArray{<:Integer,N}; epsilon=1e-6) where {T,N}
    hausdorff_dt_loss(y_pred::AbstractArray{T,N}, y::AbstractArray{Bool,N}; epsilon=1e-6) where {T,N}

Calculates the Hausdorff Distance Transform Loss.

Based on MONAI's implementation: `mean(((y_pred - y) ** 2) * dt)`
where `dt` is the distance transform of the ground truth `y`.

Args:
    y_pred: Predicted segmentation, probabilities (AbstractArray{Float, N}).
    y: Ground truth segmentation, labels/mask (AbstractArray{Integer/Bool, N}).
    epsilon: Small value added for numerical stability (though MONAI's formula doesn't seem to use it directly in the DT weighting).

Returns:
    Scalar loss value.
"""
function hausdorff_dt_loss(y_pred::AbstractArray{T,N}, y::AbstractArray{U,N}; epsilon=1e-6) where {T<:AbstractFloat, U<:Union{Integer,Bool}, N}
    if size(y_pred) != size(y)
        throw(DimensionMismatch("Prediction and ground truth arrays must have the same dimensions."))
    end
    
    backend = KernelAbstractions.get_backend(y_pred)
    
    # 1. Calculate Distance Transform of the ground truth mask y
    dt = distance_transform_ka(backend, y)
    
    # 2. Allocate memory for loss components
    loss_components = KA.zeros(backend, T, size(y_pred))
    
    # 3. Launch kernel to calculate element-wise loss components
    kernel! = hausdorff_dt_loss_kernel!(backend)
    kernel!(loss_components, y_pred, y, dt, T(epsilon), ndrange=size(y_pred))
    KernelAbstractions.synchronize(backend)
    
    # 4. Calculate the mean over all components
    loss = mean(loss_components)
    
    return loss
end


"""
    dice_loss(y_true::AbstractArray{<:Integer,N}, y_pred::AbstractArray{<:Integer,N}) where N

Calculates the Dice loss based on integer masks.
Formula: 1.0 - (2.0 * intersection / (sum(y_true) + sum(y_pred)))
"""
function dice_loss(y_true::AbstractArray{T,N}, y_pred::AbstractArray{T,N}) where {T<:Integer, N}
    intersection = sum(y_true .& y_pred)
    denom = sum(y_true) + sum(y_pred)
    
    if denom == 0
        return 0.0
    end
    
    return 1.0 - (2.0 * intersection / denom)
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
    jaccard_loss(y_true::AbstractArray{<:Integer,N}, y_pred::AbstractArray{<:Integer,N}) where N

Calculates the Jaccard Loss (1 - Jaccard Index).
"""
function jaccard_loss(y_true::AbstractArray{T,N}, y_pred::AbstractArray{T,N}) where {T<:Integer, N}
    return 1.0 - jaccard_index(y_true, y_pred)
end


"""
    cross_entropy_loss(y_true::AbstractArray{T,N}, y_pred::AbstractArray{U,N}; epsilon=1e-12) where {T<:Real, U<:AbstractFloat, N}

Calculates the binary cross-entropy loss.
Assumes y_true contains probabilities or {0, 1} values, and y_pred contains predicted probabilities.
Formula: -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
"""
function cross_entropy_loss(y_true::AbstractArray{T,N}, y_pred::AbstractArray{U,N}; epsilon=1e-12) where {T<:Real, U<:AbstractFloat, N}
    if size(y_true) != size(y_pred)
        throw(DimensionMismatch("Prediction and ground truth arrays must have the same dimensions."))
    end
    
    # Clamp predictions to avoid log(0)
    y_pred_clamped = clamp.(y_pred, U(epsilon), U(1.0 - epsilon))
    
    # Calculate element-wise loss
    loss_elements = @. - (y_true * log(y_pred_clamped) + (1 - y_true) * log(1 - y_pred_clamped))
    
    # Return the mean loss
    return mean(loss_elements)
end