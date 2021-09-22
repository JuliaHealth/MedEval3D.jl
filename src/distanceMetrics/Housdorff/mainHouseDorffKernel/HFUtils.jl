
"""
utility functions for Housedorff kernel
"""
module HFUtils
  

"""
we use it to access pading in the shmem
change the indexing so  instead of transverse plane of threads we will get coronal
    so we supply idx and idx y earlier z was constant and x,y changed so we considered top with z =1 and bottom z = 34
    now we think x and y as would be displayed on transverse plane so in order to get anterior and posterior padding plane  we need to keep y as contant 1 or 33 and rest changing
    and in order to get left and right we need to  keep x as contant to 1 or 34 (34 in case we have size of data block = 32)
    so we will access shmem in changed indexing
"""
function transvarseToCoronalThreadPlane(constantNumb::UInt8,shmem)::Bool
    return shmem[threadIdx().x,constantNumb, threadIdx().y]
end#transvarseToCoronalThreadPlane    

"""
we use it to access pading in the shmem
change the indexing so  instead of transverse plane of threads we will get coronal
    so we supply idx and idx y earlier z was constant and x,y changed so we considered top with z =1 and bottom z = 34
    now we think x and y as would be displayed on transverse plane so in order to get anterior and posterior padding plane  we need to keep y as contant 1 or 33 and rest changing
    and in order to get left and right we need to  keep x as contant to 1 or 34 (34 in case we have size of data block = 32)
    so we will access shmem in changed indexing
"""
function transvarseToSaggitalhreadPlane(constantNumb::UInt8,shmem)::Bool
    return shmem[constantNumb,threadIdx().x, threadIdx().y]
end#transvarseToSaggitalhreadPlane    


"""
clear main part of shared memory - padding will be cleared separately
"""
function clearMainShmem(shmem,zDim::UInt8)
    @unroll for zIter in UInt16(1):zDim# most outer loop is responsible for z dimension
        shmem[threadIdx().x+1,threadIdx().y+1,zIter+1]=0
    end#for 
end#clearMainShmem


"""
set padding planes to 0 
"""
function clearPadding(shmem,zDim::UInt16)
    clearHalfOfPadding(shmem,1)
    clearHalfOfPadding(shmem,zDim+2)
end#clearPadding
"""
helper for clearPadding
"""
function clearHalfOfPadding(shmem,constantNumb::UInt8)
    shmem[constantNumb,threadIdx().x, threadIdx().y]
    shmem[threadIdx().x,constantNumb, threadIdx().y]
    shmem[threadIdx().x,threadIdx().y, constantNumb]
end   


end#HFUtils