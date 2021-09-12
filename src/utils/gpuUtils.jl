module GPUutils
using CUDA

export defineIndicies,computeBlocksFromOccupancy,reduce_warp,getKernelContants

export @unroll
"""
Type{maskNumb}  - type of the numbers hold in mask
G - 3 dimensional array holding ground truth segmentation
T - 3 dimensional array holding segmentation that we want to compare to ground truth
isVariedSlice - if true it will mark that  slices have varying thickness  - hence we will need later corrextion...

"""
function defineBlocks(::Type{maskNumb} 
                    ,G::Array{maskNumb, 3}
                    ,T::Array{maskNumb, 3}
                    ,isVariedSlice::Bool
                    ) where{maskNumb}


end#defineBlocks

"""
defining basic indicies for 3 dimensional case
"""
function defineIndicies()
    i= (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    z = (blockIdx().z-1) * blockDim().z + threadIdx().z  
    return (i,j,z)

end#defineIndicies

"""
calculates for getBlockTpFpFn optimal number of blocks and thread blocks
    also it poins out to maximum number of blocks that we can squeeze on device ..
args - tupple with arguments for kernel
int32Shemm per warp - we are assuming we get some shared memory and some number of it per warp
    """
function computeBlocksFromOccupancy(args, int32Shemm)
    wanted_threads =1000000
    function compute_threads(max_threads)
        if wanted_threads > max_threads
            true ? prevwarp(device(), max_threads) : max_threads
        else
            wanted_threads
        end
    end
    compute_shmem(threads) = Int64((threads/32)*int32Shemm*sizeof(Int32) )
    
       kernel = @cuda launch=false getBlockTpFpFn(args...) 
       kernel_config = launch_configuration(kernel.fun; shmem=compute_shmem∘compute_threads)
       blocks =  kernel_config.blocks
       threads =  kernel_config.threads
       maxBlocks = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    
return blocks,threads,maxBlocks
end




"""
copied from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/perf/volumerhs.jl
"""
function loopinfo(name, expr, nodes...)
    if expr.head != :for
        error("Syntax error: pragma $name needs a for loop")
    end
    push!(expr.args[2].args, Expr(:loopinfo, nodes...))
    return expr
end

"""
copied from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/perf/volumerhs.jl
"""
macro unroll(expr)
    expr = loopinfo("@unroll", expr, (Symbol("llvm.loop.unroll.full"),))
    return esc(expr)
end


"""
Reduce a value across a warp
"""
@inline function reduce_warp( vall, lanesNumb)
    offset = UInt32(1)
    while(offset <lanesNumb) 
        vall+=shfl_down_sync(FULL_MASK, vall, offset)  
        offset<<= 1
    end
    return vall
end


"""
generally  we want to get one block per slice as dimensions of slices in the medical images are friendly - so 256x256 ; 512x512 and 1024x1024 - it should all be possible
In order to avoid edge cases we will keep number of threads to some even multiply of 32 for ecxample 512
    arguments
        threadnum - number of threads per block
        slicesNumber - number of slices of our data
        pixelNumberPerSlice - number of pixels in a slice
    output
        blockNum - number of blocks we need  - generally will return number of slices
        loopNumb - number of  iteration of each single line it need to do so a single block will cover whole slice
        indexCorr - as one lane will get access to multiple data elements we need to take correction for it 
"""
function getKernelContants(threadnum::Int,pixelNumberPerSlice::Int  )

indexCorr =  Int64(ceil(pixelNumberPerSlice/threadnum))
loopNumb= Int64(indexCorr-1)

return ( loopNumb, indexCorr )
end#getKernelContants


end #GPUutils