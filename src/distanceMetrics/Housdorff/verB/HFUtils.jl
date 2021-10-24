
"""
utility functions for Housedorff kernel
"""
module HFUtils
using Main.CUDAGpuUtils, Logging,StaticArrays
 using CUDA, Main.BasicStructs, Logging
 using Main.CUDAGpuUtils ,Main.IterationUtils
export    clearLocArr,clearMainShmem,clearPadding
export @iter3dOuter,@iterDataBlock


"""
clear main part of shared memory - padding will be cleared separately
"""
function clearMainShmem(shmem)
    @unroll for zIter in UInt8(1):UInt8(32)# most outer loop is responsible for z dimension
        shmem[threadIdxX()+1,threadIdxY()+1,zIter+1]=0
    end#for 
end#clearMainShmem

"""
clear source shmem in shared memory """
function clearSourceShmem(shmem)
    @unroll for zIter in UInt8(1):UInt8(32)
         shmem[threadIdxX(),threadIdxY(),zIter]=0
    end#for 
end#clearMainShmem


"""
specialization of 2 dim iteration  for iterating over 3 dimensional metadata
    we join y and z iterations in order to increase occupancy for potentially small arrays
"""
macro iter3dOuter(metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta, ex)
    mainExp = generalizedItermultiDim(;xname=:(xMeta)
    ,yname= :(yzSpot)
    ,arrDims=metaDataDims
    ,loopXdim=loopXMeta 
    ,loopYdim=loopYZMeta
    ,isFullBoundaryCheckY=false
    ,isFullBoundaryCheckX =false
    ,yOffset = :(ydim*gridDim().x)
    ,yAdd=  :(blockIdxX()-1) 
    ,additionalActionBeforeY= :( yMeta= rem(yzSpot,$metaDataDims[2]) ; zMeta= fld(yzSpot,$metaDataDims[2]) )
    ,yCheck = :(yMeta < $metaDataDims[2] && zMeta<$metaDataDims[3] )
    ,xCheck = :(xMeta < $metaDataDims[1])
    #so each block will iterate over all xses
    ,xOffset= :(0)
    ,xAdd= :(xdim)
    ,is3d = false
    , ex = ex)  
    return esc(:( $mainExp))

end #iter3dOuter

"""
will enable iterating over the data of data block
"""
macro iterDataBlock(mainArrDims,dataBlockDims,loopXdim ,loopYdim,loopZdim,ex)
    mainExp = generalizedItermultiDim(;arrDims=mainArrDims
    ,loopXdim
    ,loopYdim
    ,loopZdim
    # ,xCheck = :((xMeta* $dataBlockDims[1]+x)<=$mainArrDims[1] )
    # ,yCheck = :((yMeta* $dataBlockDims[2]+y)<=$mainArrDims[2])
    # ,zCheck = :( (zMeta* $dataBlockDims[3]+z)<=$mainArrDims[3])
    ,xCheck = :(((xdim * blockDimX())+threadIdxX()  )<= $dataBlockDims[1] && x<=$mainArrDims[1] )
    ,yCheck = :(((ydim * blockDimY())+threadIdxY()  )<= $dataBlockDims[2] &&  y<=$mainArrDims[2])
    ,zCheck = :((zdim+1)<= $dataBlockDims[3]  &&   z<=$mainArrDims[3])
    ,zOffset= :(zMeta* ( ($dataBlockDims[3])  ) )
    ,zAdd =:(zdim+1)
   ,yOffset = :(ydim* blockDimY()+yMeta* $dataBlockDims[2])
   ,yAdd= :(threadIdxY())
   ,xOffset= :( (xdim * blockDimX()) +xMeta* $dataBlockDims[1])
    ,xAdd= :(threadIdxX())
    ,isFullBoundaryCheckX =true
    , isFullBoundaryCheckY=true
    , isFullBoundaryCheckZ=true
    ,additionalActionBeforeX=  :(xpos= xdim * blockDimX()+threadIdxX() ;ypos= (ydim * blockDimY())+threadIdxY() ;zpos=(zdim+1))
    , ex = ex)  
    return esc(:( $mainExp))
end


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


"""
this macro will be used to 
    - decide wheather the part of the code will be executed at all - in order to avoid branching at runtime
    - select at compile time appropriate target for the statement (generally the global memory array )
    - all will be based on the  ConfigurtationStruct struct - conf
    entryName - will tell us what particularly we should look for in the conf
    if the entry of name entryName in conf will be true  we will return expression otherwise we will return nothing
    we will
    metricsArr - will be used from caller scope - and it is an array in global memory that will be      
"""
macro isInConf(conf, entryName, ex)
    
    index = 0
    if(conf.sliceWiseMatrics)
        index=3# 3 becouse one will be added in a moment
    end    
    # we look for the all of the properties that are true - we need to also remember that in case of sliceWiseMatrics - it will mark that we need 4 arrays
    for i in propertynames(eval(conf))
        if getproperty(conf, i)
            index+=1
        end#if
        if i == Symbol(entryName)
            break
        end        
    end#for   

    if getproperty(conf, Symbol(entryName))

    return esc(quote   
    
        @inbounds metricsArr[$index]  = $ex # @inbounds fp[]+= 
   
    end#quote
    )
end#if
    

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
