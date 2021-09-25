
"""
utility functions for Housedorff kernel
"""
module HFUtils
using CUDA, Main.CUDAGpuUtils, Logging,StaticArrays

export    clearLocArr,clearMainShmem,clearPadding


"""
clear main part of shared memory - padding will be cleared separately
"""
function clearMainShmem(shmem)
    @unroll for zIter in UInt8(1):UInt8(32)# most outer loop is responsible for z dimension
        shmem[threadIdxX()+1,threadIdxY()+1,zIter+1]=0
    end#for 
end#clearMainShmem

"""
clear local array
"""
function clearLocArr(locArr)
    @unroll for i in UInt8(1):UInt8(32)# most outer loop is responsible for z dimension
        locArr[i]=0
    end#for 
end#clearMainShmem

"""
set padding planes to 0 
"""
function clearPadding(shmem)
    clearHalfOfPadding(shmem,UInt8(1))
    clearHalfOfPadding(shmem,UInt8(34))
end#clearPadding
"""
helper for clearPadding
"""
function clearHalfOfPadding(shmem,constantNumb::UInt8)
    shmem[constantNumb,threadIdxX()+1, threadIdxY()+1]=false
    shmem[threadIdxX()+1,constantNumb, threadIdxY()+1]=false
    shmem[threadIdxX()+1,threadIdxY()+1, constantNumb]=false
end   





end#HFUtils

# """
# we use it to access pading in the shmem
# change the indexing so  instead of transverse plane of threads we will get coronal
#     so we supply idx and idx y earlier z was constant and x,y changed so we considered top with z =1 and bottom z = 34
#     now we think x and y as would be displayed on transverse plane so in order to get anterior and posterior padding plane  we need to keep y as contant 1 or 33 and rest changing
#     and in order to get left and right we need to  keep x as contant to 1 or 34 (34 in case we have size of data block = 32)
#     so we will access shmem in changed indexing
# """
# function transvarseToCoronalThreadPlane(constantNumb::UInt8,shmem)::Bool
#     return shmem[threadIdxX(),constantNumb, threadIdxY()]
# end#transvarseToCoronalThreadPlane    

# """
# we use it to access pading in the shmem
# change the indexing so  instead of transverse plane of threads we will get coronal
#     so we supply idx and idx y earlier z was constant and x,y changed so we considered top with z =1 and bottom z = 34
#     now we think x and y as would be displayed on transverse plane so in order to get anterior and posterior padding plane  we need to keep y as contant 1 or 33 and rest changing
#     and in order to get left and right we need to  keep x as contant to 1 or 34 (34 in case we have size of data block = 32)
#     so we will access shmem in changed indexing
# """
# function transvarseToSaggitalhreadPlane(constantNumb::UInt8,shmem)::Bool
#     return shmem[constantNumb,threadIdxX(), threadIdxY()]
# end#transvarseToSaggitalhreadPlane    
