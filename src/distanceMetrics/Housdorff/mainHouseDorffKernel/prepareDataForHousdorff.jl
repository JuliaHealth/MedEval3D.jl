using KernelAbstractions
using StaticArrays

"""
we need to give back number of false positive and false negatives and min,max x,y,x of block containing all data 
IMPORTANT - in order to avoid bound checking on every iteration we need to keep the dimension of the resulting block be divided by data block cube size for example 32
IMPORTANT - we assume that x dim can not be bigger than 1024
adapted from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/src/mapreduce.jl
goldBoolGPU3d - array holding 3 dimensional data of gold standard bollean array
segmBoolGPU3d - array with 3 dimensional the data we want to compare with gold standard
reducedGold - the smallest boolean block (3 dim array) that contains all positive entris from both masks
reducedSegm - the smallest boolean block (3 dim array) that contains all positive entris from both masks
numberToLooFor - number we will analyze whether is the same between two sets
loopNumbYdim - number of times the single lane needs to loop in order to get all needed data - in this kernel it will be exactly a y dimension of a slice
xdim - length in x direction of source array 
loopNumbXdim - in case the x dim will be bigger than number of threads we will create second inner loop
cuda arrays holding just single value wit atomically reduced result
,fn,fp
,minxRes,maxxRes
,minyRes,maxyRes
,minZres,maxZres
"""
@kernel function getBoolCubeKernel(goldBoolGPU3d, segmBoolGPU3d, reducedGoldA, reducedSegmA, reducedGoldB, reducedSegmB,
                                     loopNumbYdim::UInt16, xdim::UInt16, loopNumbXdim::UInt16, numberToLooFor,
                                     IndexesArray, fn, fp, minxRes, maxxRes, minyRes, maxyRes, minZres, maxZres, warpNumber)
    # Local variables
    anyPositive = false # True if any bit is positive in this array
    locArr = zeros(MVector{6, UInt16}) # Local array for temporary results

    # Shared memory initialization
    shmemSum = zeros(Float32, warpNumber) # Replace with KernelAbstractions-compatible shared memory

    # Loop through 3D grid
    for z in 1:size(goldBoolGPU3d, 3)
        for y in 1:size(goldBoolGPU3d, 2)
            for x in 1:size(goldBoolGPU3d, 1)
                # Check if the current element matches the target number
                if @inbounds goldBoolGPU3d[x, y, z] == numberToLooFor
                    # Update local variables
                    locArr[1] += UInt16(1) # Example: increment false negatives
                    locArr[2] += UInt16(1) # Example: increment false positives
                    locArr[3] = min(locArr[3], UInt16(x)) # Update min x
                    locArr[4] = max(locArr[4], UInt16(x)) # Update max x
                    locArr[5] = min(locArr[5], UInt16(y)) # Update min y
                    locArr[6] = max(locArr[6], UInt16(y)) # Update max y
                end
            end
        end

        # Synchronize threads
        @sync_threads

        # Reduce results across threads
        for i in 1:warpNumber
            shmemSum[i] += locArr[1] # Example: reduce false negatives
        end

        # Synchronize threads again
        @sync_threads
    end

    # Write results back to global memory
    fn[1] = shmemSum[1] # Example: write reduced false negatives
    fp[1] = shmemSum[2] # Example: write reduced false positives
    minxRes[1] = locArr[3]
    maxxRes[1] = locArr[4]
    minyRes[1] = locArr[5]
    maxyRes[1] = locArr[6]
    minZres[1] = locArr[3] # Example: write min z
    maxZres[1] = locArr[4] # Example: write max z
end
