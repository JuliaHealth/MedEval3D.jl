
"""
initializations of Housedorff kernelalso clear up functions
"""
module HKernelInits
using CUDA, ..CUDAGpuUtils, Logging,StaticArrays
using ..HFUtils
export memoryAllocations,clearMainShmem,clearPadding,clearHalfOfPadding

"""
initialize constants
"""
function memoryAllocations()
    #for storing results
    #shmemGold,shmemSegm,shmemSum = createAndInitializeShmem(datablockDim, threadIdxX())
    resShmem =  @cuStaticSharedMem(Bool,(34,34,34))#+2 in order to get the one padding 
    #we need to set the values of the shared memory to 0 as it is not guaranteed by CUDA to be so 
    clearMainShmem(resShmem,32)
    clearPadding(resShmem,32)# we separately clear padding
    #coordinates of data in main array
    #we will use this to establish weather we should mark  the data block as empty or full ...
    isMaskFull= zeros(MVector{1,Bool})
    isMaskEmpty= ones(MVector{1,Bool}) 
    #here we will store in registers data uploaded from mask for later verification wheather we should send it or not
    locArr= zeros(MVector{32,Bool})

    return(resShmem,isMaskFull, isMaskEmpty,locArr)

end#initializeMemory

end#HKernelInits