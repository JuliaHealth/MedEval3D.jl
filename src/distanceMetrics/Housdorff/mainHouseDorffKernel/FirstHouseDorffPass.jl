"""
we already converted source array to booleans now we know the number of false positives; false negatives and 
min,max x,y,z of part of data where we have any intreasting maxresultPoints

Now we need to: 
    -populate metadata
        -mark also whether the block is active, foll or neither - from both perspectives of both masks
        -we can set the min,max dims of data block
    -as we already loaded data from global memory - what is supposedly most expensive part - we can do the first dilatation step
    -we prepare rudamentary work plan for  next step    


Toexperiment
    reducePaddingPlane   #to be experimented is it better to reduce to varyiong y or x 



"""
module FirstHouseDorffPass
using CUDA, Main.GPUutils, Logging,StaticArrays

"""
prepare first pass of Housedorff kernel run
datablockDim - the edge length of cube describing the block of data we are working on  - defoult is 32 - which would produce blocks of size 32x32x32
"""
function executefirstHouseDorffPassKernel(datablockDim,metaData)

    threadsNum = (datablockDim,datablockDim)
    blocksNum = size(metaData)



end#executefirstHouseDorffPassKernel
"""


main kernel of muodule it will get 4 data arrays -   reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB
    we work on all simultaneously
    metaData - basic data about data blocks - stored in 3 dimensional array
    metadataDims - dimensions of metaData array
    resArrayA, resArrayB- 3 dimensional array of size of reduced source array where entry will be in place where we had covered the voxel, and value will mark in which itwration
        - we have two result arrays separate for each pass    
        - when we will cover some point  we need to report it in the resArray
    localResLastEntryList- results that will be managed by given thread block
    workSchedule - on the basis of it next  phase thread blocks will know on what data blocks to work
    zDim - dimension of a block in z direction - controls the number of iterations we will do by the thread plane
    datablockDim - the edge length  describing the block of data we are working on  - defoult is 32 - which would produce blocks of size 32x32xzDim
        - we will have  the plane of threads of size 32x32 and in order to preserve memory coalescence we should have a warp have the  set of next x positions
    mainArrayDims - dimensions of the main array needed to check for boundary conditions     

"""
function firstHouseDorffPassKernel(  reducedGoldA
                                    ,reducedSegmA
                                    ,reducedGoldB
                                    ,reducedSegmB
                                    ,metaData
                                    ,metadataDims
                                    ,resArrayA,resArrayB 
                                    ,localResLastEntryList
                                    ,workSchedule
                                    ,datablockDim 
                                    ,mainArrayDims
                                    ,zDim::UInt16 )
    #constant for all threads in a block
    iterationNumber= UInt16(0)

    ############initializations of data 
    resShmem,  isMaskFull, isMaskEmpty,locArr = memoryAllocations(datablockDim,zDim)

    x,y,z krowa

    ############### execution
    executeDataIter(zDim,analyzedArr, refAray,iterationNumber,x,y,z)

    #now we need to deal with padding in shmem res
    processAllPaddingPlanes(x,y,z,shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr,metaData,metadataDims,isPassGold,locArr)


    # now let's check weather block is eligible for futher processing - for this we need sums ...
    #first we check eligibity from gold mask perspective    
     no!! we will modify reduce so it will work on bitwise operators not adding @inbounds shmemSum[wid] = reduce_warp(bitGold,32)

end #FirstHouseDorffPassKernel




"""
return basic x,y,z coordinates of thread; z needs to be adjusted for streaming ...

we put it into separate function mainly in order to  cast it to UInt16
"""
function getxyz()::Tuple{UInt16,UInt16,UInt16}
return ((blockIdx().x-1)*32+1,(blockIdx().x-1)*32+1, (blockIdx().z-1)*32+1)
end#getxyz










end #FirstHouseDorffPass


# """
# creates shared memory and initializes it to 0
# wid - the number of the warp in the block
# """
# function createAndInitializeShmem(datablockDim, threadIdX)
#    shmemGold = @cuStaticSharedMem(Bool, (datablockDim,datablockDim,datablockDim))
#    shmemSegm = @cuStaticSharedMem(Bool, (datablockDim,datablockDim,datablockDim))
#    #in order to help with summing operations - so we will know wheather block is empty /full ...
#    shmemSums = @cuStaticSharedMem(UInt8,datablockDim)
#    shmemSums[threadIdX] =0
# return (shmemGold,shmemSegm,shmemSums)
# end#createAndInitializeShmem
