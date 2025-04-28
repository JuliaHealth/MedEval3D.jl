using CUDA
using KernelAbstractions
using Atomix  # For atomic operations on GPU

backend = CUDABackend()

# to match the dimensions of the mon ai library
function permute_data_format(x)
    dims = ndims(x)
    if dims == 4  # 2D case
        return permutedims(x, (4, 1, 2, 3))  # (C,H,W,N) -> (N,C,H,W)
    elseif dims == 5  # 3D case
        return permutedims(x, (5, 1, 2, 3, 4))  # (C,D,H,W,N) -> (N,C,D,H,W)
    end
end

@kernel function reduce_kernel_final!(input1, input2, output, total_elements)
    # CUDA thread and block indexing
    block_idx = (blockIdx().x - 1) * blockDim().x
    idx = block_idx + threadIdx().x
    tid = threadIdx().x

    # Allocate shared memory for partial sums
    local_sums = @localmem(Float32, 256)  # 256 = threads per block

    # Initialize local shared memory
    local_sums[tid] = 0.0f0

    @synchronize

    # Stride across array
    stride = blockDim().x * gridDim().x

    dims = ndims(input1)

    for i in idx:stride:total_elements
        if i <= total_elements
            val = input1[i] * input2[i]
            local_sums[tid] += val
        end
    end

    @synchronize

    # Reduction within block using manual unroll
    for s in (128, 64, 32, 16, 8, 4, 2, 1)
        if tid <= s
            local_sums[tid] += local_sums[tid + s]
        end
        @synchronize
    end

    # Only thread 1 updates the final output atomically
    if tid == 1
        Atomix.@atomic output[1] += local_sums[1]
    end
end

@kernel function sum_kernel_final!(input, output, total_elements)
    # Thread/block index
    block_idx = (blockIdx().x - 1) * blockDim().x
    idx = block_idx + threadIdx().x
    tid = threadIdx().x

    # Shared memory per block
    local_sums = @localmem(Float32, 256)

    local_sums[tid] = 0.0f0

    @synchronize

    stride = blockDim().x * gridDim().x

    for i in idx:stride:total_elements
        if i <= total_elements
            local_sums[tid] += input[i]
        end
    end

    @synchronize

    for s in (128, 64, 32, 16, 8, 4, 2, 1)
        if tid <= s
            local_sums[tid] += local_sums[tid + s]
        end
        @synchronize
    end

    if tid == 1
        Atomix.@atomic output[1] += local_sums[1]
    end
end

# cross entropy logic
@kernel function cross_entropy_kernel_final!(input, target, output, epsilon, total_elements)
    block_idx = (blockIdx().x - 1) * blockDim().x
    idx = block_idx + threadIdx().x
    tid = threadIdx().x

    # Allocate shared memory
    local_ce = @localmem(Float32, 256)
    local_ce[tid] = 0.0f0

    @synchronize

    stride = blockDim().x * gridDim().x

    for i in idx:stride:total_elements
        if i <= total_elements
            @inbounds begin
                pred = clamp(input[i], epsilon, 1.0f0 - epsilon)
                target_val = target[i]
                ce = -target_val * log(pred) - (1.0f0 - target_val) * log(1.0f0 - pred)
                local_ce[tid] += ce
            end
        end
    end

    @synchronize

    # Reduce within the block
    for s in (128, 64, 32, 16, 8, 4, 2, 1)
        if tid <= s
            local_ce[tid] += local_ce[tid + s]
        end
        @synchronize
    end

    # Only thread 1 writes the final result
    if tid == 1
        Atomix.@atomic output[1] += local_ce[1]
    end
end

function dice_loss(input::CuArray{Float32}, target::CuArray{Float32}; epsilon=1e-5, sigmoid=true)
    @assert ndims(input) in (4, 5) "Input must be 4D or 5D"
    @assert size(input) == size(target) "Input and target must have the same shape"

    input = permute_data_format(input)
    target = permute_data_format(target)

    if sigmoid
        input .= 1.0 ./ (1.0 .+ exp.(-input))
    end

    # Total number of elements
    total_elements = prod(size(input))

    # Allocate GPU accumulators
    intersection = CUDA.zeros(Float32, 1)
    sum_input    = CUDA.zeros(Float32, 1)
    sum_target   = CUDA.zeros(Float32, 1)

    # Configure kernel launch
    threads_per_block = 256
    blocks = cld(total_elements, threads_per_block)

    # Launch GPU kernels
    reduce_kernel_final!(backend)(
        input, target, intersection, total_elements;
        ndrange = blocks * threads_per_block,
        workgroupsize = threads_per_block
    )

    sum_kernel_final!(backend)(
        input, sum_input, total_elements;
        ndrange = blocks * threads_per_block,
        workgroupsize = threads_per_block
    )

    sum_kernel_final!(backend)(
        target, sum_target, total_elements;
        ndrange = blocks * threads_per_block,
        workgroupsize = threads_per_block
    )

    KernelAbstractions.synchronize(backend)

    # Safely retrieve scalar values
    numerator = 2.0f0 * CUDA.@allowscalar intersection[1] + epsilon
    denominator = CUDA.@allowscalar sum_input[1] + CUDA.@allowscalar sum_target[1] + epsilon
    dice = numerator / denominator

    return 1.0f0 - dice
end

function jaccard_loss(input::CuArray{Float32}, target::CuArray{Float32}; epsilon=1e-5, sigmoid=true)
    @assert ndims(input) in (4, 5) "Input must be 4D or 5D"
    @assert size(input) == size(target) "Input and target must have same size"

    input = permute_data_format(input)
    target = permute_data_format(target)

    if sigmoid
        input .= 1.0 ./ (1.0 .+ exp.(-input))
    end

    total_elements = prod(size(input))

    # Allocate GPU accumulators
    intersection = CUDA.zeros(Float32, 1)
    union_sum    = CUDA.zeros(Float32, 1)

    # Prepare union array: input + target - (input * target)
    union_input = CUDA.zeros(Float32, size(input))
    @. union_input = input + target - (input * target)

    # Kernel config
    threads_per_block = 256
    blocks = cld(total_elements, threads_per_block)

    # Launch kernels
    reduce_kernel_final!(backend)(
        input, target, intersection, total_elements;
        ndrange = blocks * threads_per_block,
        workgroupsize = threads_per_block
    )

    sum_kernel_final!(backend)(
        union_input, union_sum, total_elements;
        ndrange = blocks * threads_per_block,
        workgroupsize = threads_per_block
    )

    KernelAbstractions.synchronize(backend)

    # Safely read scalars
    intersection_val = CUDA.@allowscalar intersection[1]
    union_val = CUDA.@allowscalar union_sum[1]

    jaccard = (intersection_val + epsilon) / (union_val + epsilon)

    return 1.0f0 - jaccard
end


function cross_entropy_loss(input::CuArray{Float32}, target::CuArray{Float32}; epsilon=1e-5, sigmoid=true)
    @assert ndims(input) in (4, 5) "Input must be 4D or 5D"
    @assert size(input) == size(target) "Input and target must have same size"

    input = permute_data_format(input)
    target = permute_data_format(target)

    if sigmoid
        input .= 1.0f0 ./ (1.0f0 .+ exp.(-input))
    end

    total_elements = prod(size(input))

    # Output accumulator
    ce_sum = CUDA.zeros(Float32, 1)

    # Kernel configuration
    threads_per_block = 256
    blocks = cld(total_elements, threads_per_block)

    # Launch the cross entropy kernel
    cross_entropy_kernel_final!(backend)(
        input, target, ce_sum, epsilon, total_elements;
        ndrange = blocks * threads_per_block,
        workgroupsize = threads_per_block
    )

    KernelAbstractions.synchronize(backend)

    # Safely read and normalize
    total_ce = CUDA.@allowscalar ce_sum[1]
    return total_ce / Float32(total_elements)
end