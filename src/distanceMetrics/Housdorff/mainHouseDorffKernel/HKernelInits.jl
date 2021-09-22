
"""
initializations of Housedorff kernelalso clear up functions
"""
module HKernelInits
using CUDA, Main.GPUutils, Logging,StaticArrays
using Main.HFUtils
export memoryAllocations,clearMainShmem,clearPadding,clearHalfOfPadding

"""
initialize constants
zDim - how many times we stream across third dimension
"""
function memoryAllocations(datablockDim::UInt16,zDim::UInt16)
    #for storing results
    #shmemGold,shmemSegm,shmemSum = createAndInitializeShmem(datablockDim, threadIdx().x)
    resShmem = CuStaticSharedArray(Bool,(datablockDim+2,datablockDim+2,zDim +2))#+2 in order to get the one padding 
    #we need to set the values of the shared memory to 0 as it is not guaranteed by CUDA to be so 
    clearMainShmem(resShmem,zDim )
    clearPadding(resShmem,zDim)# we separately clear padding
    #coordinates of data in main array
    #we will use this to establish weather we should mark  the data block as empty or full ...
    isMaskFull= zeros(MVector{1,Bool})
    isMaskEmpty= ones(MVector{1,Bool}) 
    #here we will store in registers data uploaded from mask for later verification wheather we should send it or not
    locArr= zeros(MVector{zDim,Bool})

    return(resShmem,isMaskFull, isMaskEmpty,locArr)

end#initializeMemory






end#HKernelInits