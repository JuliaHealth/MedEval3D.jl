module CUDAGpuUtils


using CUDA, StaticArrays

export gridDimX,atomicallySetValueTrreeDim,atomicallyAddOneInt,clearLocArrdefineIndicies,computeBlocksFromOccupancy,reduce_warp,getKernelContants,assignWorkToCooperativeBlocks,getMaxBlocksPerMultiproc,reduce_warp_max,reduce_warp_min,reduce_warp_min,reduce_warp_or,reduce_warp_and,blockIdxZ,blockIdxY,blockIdxX,blockDimZ, blockDimY, blockDimX, threadIdxX, threadIdxY, threadIdxZ
export @unroll, @ifX, @ifY, @ifXY, @widL, @wid, @lan


"""
convinience macro that will execute only if it has given thread Id X
"""
macro ifX(x, ex)
    return esc(:(if threadIdxX()==$x
        $ex
    end))

end

"""
convinience macro that will execute only if it has given thread Id Y
"""
macro ifY(y, ex)
    return esc(:(if threadIdxY()==$y 
        $ex
    end))

end

"""
convinience macro that will execute only if it has given thread Id Y
"""
macro ifXY(x,y, ex)
        return esc(:(if threadIdxY()==$y && threadIdxX()==$x
            $ex
        end))

end
"""
convinience macro that will execute only if it has given wid and lane 
"""
macro widL(innerWid,innerLane, ex)
    return esc(:(if wid==$innerWid && lane==$innerLane
        $ex
    end))

end



"""
convinience macro that will execute only if it has given wid 
"""
macro wid(innerWid, ex)
    return esc(:(if wid==$innerWid 
        $ex
    end))

end
"""
convinience macro that will execute only if it has given lane 
"""
macro lan(innerWid,innerLane, ex)
    return esc(:(if lane==$innerLane
        $ex
    end))

end


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
wrapper to get x coordinates of thread
"""
function threadIdxX()::UInt32
    threadIdx().x
end
"""
wrapper to get y coordinates of thread
"""
function threadIdxY()::UInt32
    threadIdx().y
end
"""
wrapper to get x coordinates of thread
"""
function threadIdxZ()::UInt32
    threadIdx().z
end






"""
wrapper to get x dimension of thread block
"""
function blockDimX()::UInt32
    blockDim().x
end
"""
wrapper to get y dimension of thread block
"""
function blockDimY()::UInt32
    blockDim().y
end
"""
wrapper to get x dimension of thread block
"""
function blockDimZ()::UInt32
    blockDim().z
end



"""
wrapper to get x coordinates of thread block
"""
function blockIdxX()::UInt32
    blockIdx().x
end
"""
wrapper to get y coordinates of thread block
"""
function blockIdxY()::UInt32
    blockIdx().y
end
"""
wrapper to get x coordinates of thread block
"""
function blockIdxZ()::UInt32
    blockIdx().z
end

"""
wrapper to get x length of grid - how many thread blocks in x dimension
"""
function gridDimX()::UInt32
    gridDim().x
end







"""
defining basic indicies for 3 dimensional case
"""
function defineIndicies()
    i= (blockIdx().x-1) * blockDim().x + threadIdxX()
    j = (blockIdx().y-1) * blockDim().y + threadIdxY()
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
Reduce a value across a warp and sum
"""
@inline function reduce_warp( vall::T, lanesNumb) where T
        
    offset = UInt16(1)
    while(offset <lanesNumb) 
        vall+=shfl_down_sync(FULL_MASK, vall, offset)  
        offset<<= 1
    end

    return vall
end

"""
Reduce a value across a warp and return max
"""
@inline function reduce_warp_max( vall, lanesNumb::UInt8)
    offset = UInt16(1)
    while(offset <lanesNumb) 
        vall=max(vall, shfl_down_sync(FULL_MASK, vall, offset))
        offset<<= 1
    end
    return vall
end


"""
Reduce a value across a warp and return min 
"""
@inline function reduce_warp_min( vall::T, lanesNumb::UInt8) where T
    offset = UInt16(1)
    while(offset <lanesNumb) 
        #if(vall==0)  vall=T(10000) end
        vall=min(vall, shfl_down_sync(FULL_MASK, vall, offset))
        offset<<= 1
    end
    return vall
end


"""
Reduce a value across a warp and return or (so if any value is true it will return true) 
"""
@inline function reduce_warp_or( vall::Bool, lanesNumb::UInt8) 
    offset = UInt16(1)
    while(offset <lanesNumb) 
        #if(vall==0)  vall=T(10000) end
        vall=vall | shfl_down_sync(FULL_MASK, vall, offset)
        offset<<= 1
    end
    return vall
end

"""
Reduce a value across a warp and return and (so if all values are true it will return true) 
"""
@inline function reduce_warp_and( vall::Bool, lanesNumb::UInt8) 
    offset = UInt16(1)
    while(offset <lanesNumb) 
        #if(vall==0)  vall=T(10000) end
        vall=vall & shfl_down_sync(FULL_MASK, vall, offset)
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



"""
Calculates  maximum number of 
args - tuple with arguments needed to execute a kernel (it will not be executed it is only needed for occupancy API)
kernelFun - the function describing kernel

"""
function getMaxBlocksPerMultiproc(args,kernelFun )
 kernel = @cuda launch=false kernelFun(args...) 
return CUDA.active_blocks(kernel.fun, 512)  
end#getMaxBlocksPerMultiproc


"""
now plan is to create a matrix on the CPU that will be pushed onto the GPU
matrix will tell each block which slices it should manage in order to cover all slices
number of blocks will be calclulated in a way to fit them all in a single grid and enable full utilization of cooperative groups 
matrix will be n x m  where 
    n is number of blocks we can fit in the grid
    m is maximum number of slices managed per block - of course in most cases some blocks will not be needed in final round - in this case this will evaluate to 0 ...
  code adapted from 
    https://stackoverflow.com/questions/63929929/processing-shared-work-queue-using-cuda-atomic-operations-and-grid-synchronizati/63930239#63930239
    https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#grid-synchronization-cg
    https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/test/execution.jl

    generally the number of threads to avoid corner cases  needs to be 256 or 512 or 1024

    slicesNumb - number of slices to manage, generally it is assumed that each block controls one slice
    numberOfBlocksPerMultprocessor- how many block can be run in a single SM
    """
function assignWorkToCooperativeBlocks(slicesNumb, numberOfBlocksPerMultprocessor=1 )
   numberOfBlocks = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)*numberOfBlocksPerMultprocessor
   maxSlicesPerBlock = Int64(ceil(slicesNumb/numberOfBlocks))
   sliceAssignMatrix = ones(Int8,numberOfBlocks, maxSlicesPerBlock ).-2
   # first filling first 4 columns
   for i in 0:numberOfBlocks-1
    for j in 1:maxSlicesPerBlock-1
        z = i*(maxSlicesPerBlock-1) +j
        if(z<=slicesNumb) 
          sliceAssignMatrix[i+1,j]=z           
        end#if    
    end#for 
   end#for 
   index =0
   #filling last column with what is left
   for i in (maximum(sliceAssignMatrix)+1):slicesNumb
    index+=1
    sliceAssignMatrix[index,maxSlicesPerBlock]= i
   end#for 


    zz = CuArray(sliceAssignMatrix)
   return (maxSlicesPerBlock, zz,numberOfBlocks)
end #assignWorkToCooperativeBlocks


####### profiling
#   nv-nsight-cu-cli --mode=launch julia 















end #CUDAGpuUtils