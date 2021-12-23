
"""
utility functions for Housedorff kernel
"""
module HFUtils
using Main.CUDAGpuUtils, Logging,StaticArrays
 using CUDA, Main.BasicStructs, Logging
 using Main.CUDAGpuUtils ,Main.IterationUtils
export    clearLocArr,clearMainShmem,clearPadding,getIndexOfQueue
export @iter3dOuter,@iterDataBlock , calculateLoopsIter, @iterDataBlockZdeepest




"""
invoked before kernel execution in order to set number of needed loop iterations
dataBdim - dimensions of the data block 
threadsXdim  - x dimension of the thread block
threadsYdim  - y dimension of the thread block
return tuple with numbers indicating how many iterations are needed in the loops of the kernel
"""
function calculateLoopsIter(dataBdim,threadsXdim,threadsYdim,metaDataDims,blocks)
    loopAXFixed= fld(dataBdim[2], threadsXdim)
    loopBXfixed= fld(dataBdim[3], threadsYdim)
            
    loopAYFixed= fld(dataBdim[1], threadsXdim)
    loopBYfixed= fld(dataBdim[3], threadsYdim)
            
    loopAZFixed= fld(dataBdim[1], threadsXdim)
    loopBZfixed= fld(dataBdim[2], threadsYdim)

    loopdataDimMainX = fld(dataBdim[1], threadsXdim)
    loopdataDimMainY = fld(dataBdim[2], threadsYdim)
    loopdataDimMainZ =dataBdim[2]
    inBlockLoopX,inBlockLoopY,inBlockLoopZ= (0,fld(dataBdim[2] ,threadsYdim),dataBdim[3]);
    metaDataLength= metaDataDims[1]*metaDataDims[2]*metaDataDims[3]
    
    loopMeta= fld(metaDataLength,blocks )
    loopWarpMeta= cld(metaDataLength,(blocks*threadsXdim ))
    resShmemTotalLength=(dataBdim[1]+2)*(dataBdim[2]+2)*(dataBdim[3]+2)
    sourceShmemTotalLength= dataBdim[1]*dataBdim[2]*dataBdim[3]
    clearIterResShmemLoop= fld(resShmemTotalLength,threadsXdim*threadsYdim)
    clearIterSourceShmemLoop= fld(sourceShmemTotalLength,threadsXdim*threadsYdim)

    shmemblockDataLenght = threadsXdim*threadsYdim*2
    shmemblockDataLoop = fld(shmemblockDataLenght,threadsXdim*threadsYdim)
    
    inBlockLoopXZIterWithPadding= fld(32,threadsYdim)

    return (inBlockLoopXZIterWithPadding,shmemblockDataLoop,shmemblockDataLenght,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength)
end    

"""
1)   Left FP  
2)   Left FN  
3)   Right FP  
4)   Right FN  
5)   Posterior FP  
6)   Posterior FN  
7)   Anterior FP  
8)   Anterior FN  
9)   Top FP  
10)   Top FN  
11)   Bottom FP  
12)   Bottom FN  
13)   Total block Fp  
14)   Total block Fn  

xpos,ypos,zpos -current  position in x,y,z dimension 
dataBdim - dimensions of the data block
boolSegm - true if we have true in algo arr 
on the basis of the data it should give the index from 1 to 14 - to the appropriate queue
"""
function getIndexOfQueue(xpos,ypos,zpos, dataBdim,boolSegm)
    #we need to do so many != in order to deal with corners ...
    return (
     (xpos==1)*1
    +(xpos==dataBdim[1])*3
    +(ypos==1 && xpos!=1 && xpos!=dataBdim[1] )*5
    +(ypos==dataBdim[2] && xpos!=1 && xpos!=dataBdim[1]  )*7
    +(zpos==1  && xpos!=1 && xpos!=dataBdim[1]  && ypos!=1 && ypos!=dataBdim[2] )*9
    +(zpos==dataBdim[3] && xpos!=1 && xpos!=dataBdim[1]  && ypos!=1 && ypos!=dataBdim[2])*11
    +(xpos>1 && xpos<dataBdim[1] &&  ypos>1 && ypos<dataBdim[2] && zpos>1 && zpos<dataBdim[3])*13
    )+boolSegm# in that way we will get odd for fp an even for fn

end


"""
specialization of 2 dim iteration  for iterating over 3 dimensional metadata
    we join y and z iterations in order to increase occupancy for potentially small arrays
"""
macro iter3dOuter(metaDataDims,loopMeta,metaDataLength, ex)
    
    return  esc(quote
    linIdexMeta = UInt32(0)
    @unroll for j in 0:($loopMeta-1)
        linIdexMeta= blockIdxX()+ j*gridDimX()-1
        xMeta= rem(linIdexMeta,$metaDataDims[1])
        zMeta= fld((linIdexMeta),$metaDataDims[1]*$metaDataDims[2])
        yMeta= fld((linIdexMeta-((zMeta*$metaDataDims[1]*$metaDataDims[2] ) + xMeta )),$metaDataDims[1])
      $ex
    end 
    linIdexMeta= blockIdxX()+ $loopMeta*gridDimX() -1
            if(linIdexMeta<$metaDataLength)
            xMeta= rem(linIdexMeta,$metaDataDims[1])
            zMeta= fld((linIdexMeta),$metaDataDims[1]*$metaDataDims[2])
            yMeta= fld((linIdexMeta-((zMeta*$metaDataDims[1]*$metaDataDims[2] ) + xMeta )),$metaDataDims[1])
        $ex
      end 
    end)


end #iter3dOuter

"""
will enable iterating over the data of data block
"""
macro iterDataBlock(mainArrDims,dataBlockDims,loopXdim ,loopYdim,loopZdim,xMeta,yMeta,zMeta, ex)
    mainExp = generalizedItermultiDim(;arrDims=mainArrDims
    ,loopXdim
    ,loopYdim
    ,loopZdim
    # ,xCheck = :((xMeta* $dataBlockDims[1]+x)<=$mainArrDims[1] )
    # ,yCheck = :((yMeta* $dataBlockDims[2]+y)<=$mainArrDims[2])
    # ,zCheck = :( (zMeta* $dataBlockDims[3]+z)<=$mainArrDims[3])
    ,xCheck = :(((xdim * blockDimX())+threadIdxX()) <= $dataBlockDims[1] && x<=$mainArrDims[1])
    ,yCheck = :(((ydim * blockDimY())+threadIdxY()) <= $dataBlockDims[2] && y<=$mainArrDims[2])
    ,zCheck = :((zdim+1) <= $dataBlockDims[3]  &&   z <= $mainArrDims[3])
    ,zOffset= :($zMeta* ( ($dataBlockDims[3])))
    ,zAdd =:(zdim+1)
   ,yOffset = :(ydim* blockDimY()+$yMeta* $dataBlockDims[2])
   ,yAdd= :(threadIdxY())
   ,xOffset= :( (xdim * blockDimX()) +$xMeta* $dataBlockDims[1])
    ,xAdd= :(threadIdxX())
    ,isFullBoundaryCheckX =true
    , isFullBoundaryCheckY=true
    , isFullBoundaryCheckZ=true
    ,additionalActionBeforeX=  :(xpos= xdim * blockDimX()+threadIdxX() ;ypos= (ydim * blockDimY())+threadIdxY() ;zpos=(zdim+1))
    , ex = ex)  
    return esc(:( $mainExp))
end



"""
will enable iterating over the data of data block
"""
macro iterDataBlockZdeepest(mainArrDims,dataBlockDims,loopXdim ,loopYdim,loopZdim,xMeta,yMeta,zMeta, ex,additionalActionAfterX)
    mainExp = generalizedItermultiDimZdeepest(;arrDims=mainArrDims
    ,loopXdim
    ,loopYdim
    ,loopZdim
    # ,xCheck = :((xMeta* $dataBlockDims[1]+x)<=$mainArrDims[1] )
    # ,yCheck = :((yMeta* $dataBlockDims[2]+y)<=$mainArrDims[2])
    # ,zCheck = :( (zMeta* $dataBlockDims[3]+z)<=$mainArrDims[3])
    ,xCheck = :(((xdim * blockDimX())+threadIdxX()) <= $dataBlockDims[1] && x<=$mainArrDims[1])
    ,yCheck = :(((ydim * blockDimY())+threadIdxY()) <= $dataBlockDims[2] && y<=$mainArrDims[2])
    ,zCheck = :((zdim+1) <= $dataBlockDims[3]  &&   z <= $mainArrDims[3])
    ,zOffset= :($zMeta* ( ($dataBlockDims[3])))
    ,zAdd =:(zdim+1)
   ,yOffset = :(ydim* blockDimY()+$yMeta* $dataBlockDims[2])
   ,yAdd= :(threadIdxY())
   ,xOffset= :( (xdim * blockDimX()) +$xMeta* $dataBlockDims[1])
    ,xAdd= :(threadIdxX())
    ,isFullBoundaryCheckX =true
    , isFullBoundaryCheckY=true
    , isFullBoundaryCheckZ=true
    ,additionalActionBeforeX=  :(xpos= xdim * blockDimX()+threadIdxX() ;ypos= (ydim * blockDimY())+threadIdxY())
    ,additionalActionBeforeZ=  :(zpos=(zdim+1))
    ,additionalActionAfterX=additionalActionAfterX
    , ex = ex)  
    return esc(:( $mainExp))
end

#macro iterDataBlock(mainArrDims,dataBlockDims,loopXdim ,loopYdim,loopZdim,ex)
    #     mainExp = generalizedItermultiDim(;arrDims=mainArrDims
    #     ,loopXdim
    #     ,loopYdim
    #     ,loopZdim
    #     # ,xCheck = :((xMeta* $dataBlockDims[1]+x)<=$mainArrDims[1] )
    #     # ,yCheck = :((yMeta* $dataBlockDims[2]+y)<=$mainArrDims[2])
    #     # ,zCheck = :( (zMeta* $dataBlockDims[3]+z)<=$mainArrDims[3])
    #     ,xCheck = :(((xdim * blockDimX())+threadIdxX()) <= $dataBlockDims[1] && x<=$mainArrDims[1])
    #     ,yCheck = :(((ydim * blockDimY())+threadIdxY()) <= $dataBlockDims[2] && y<=$mainArrDims[2])
    #     ,zCheck = :((zdim+1) <= $dataBlockDims[3]  &&   z <= $mainArrDims[3])
    #     ,zOffset= :((zMeta-1)* ( ($dataBlockDims[3])))
    #     ,zAdd =:(zdim+1)
    #    ,yOffset = :(ydim* blockDimY()+(yMeta-1)* $dataBlockDims[2])
    #    ,yAdd= :(threadIdxY())
    #    ,xOffset= :( (xdim * blockDimX()) +(xMeta-1)* $dataBlockDims[1])
    #     ,xAdd= :(threadIdxX())
    #     ,isFullBoundaryCheckX =true
    #     , isFullBoundaryCheckY=true
    #     , isFullBoundaryCheckZ=true
    #     ,additionalActionBeforeX=  :(xpos= xdim * blockDimX()+threadIdxX() ;ypos= (ydim * blockDimY())+threadIdxY() ;zpos=(zdim+1))
    #     , ex = ex)  
    #     return esc(:( $mainExp))
    # end
    
    


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



# """
# clear main part of shared memory - padding will be cleared separately
# """
# function clearMainShmem(shmem)
#     @unroll for zIter in UInt8(1):UInt8(32)# most outer loop is responsible for z dimension
#         shmem[threadIdxX()+1,threadIdxY()+1,zIter+1]=0
#     end#for 
# end#clearMainShmem

# """
# clear source shmem in shared memory """
# function clearSourceShmem(shmem)
#     @unroll for zIter in UInt8(1):UInt8(32)
#          shmem[threadIdxX(),threadIdxY(),zIter]=0
#     end#for 
# end#clearMainShmem


# """
# set padding planes to 0 
# """
# function clearPadding(shmem)
#     clearHalfOfPadding(shmem,UInt8(1))
#     clearHalfOfPadding(shmem,UInt8(34))
# end#clearPadding
# """
# helper for clearPadding
# """
# function clearHalfOfPadding(shmem,constantNumb::UInt8)
#     shmem[constantNumb,threadIdxX()+1, threadIdxY()+1]=false
#     shmem[threadIdxX()+1,constantNumb, threadIdxY()+1]=false
#     shmem[threadIdxX()+1,threadIdxY()+1, constantNumb]=false
# end   

