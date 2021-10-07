

module MeansMahalinobis
using Main.BasicPreds, Main.CUDAGpuUtils, CUDA, Main.IterationUtils, Main.ReductionUtils, Main.MemoryUtils
"""
IMPORTANT x dim of threadblock needs to be always 32 
goldArr,segmArr  - arrays we analyze 
numberToLooFor - number we are intrested in in the array
loopYdim, loopXdim, loopZdim - number of times we nned to loop over those dimensions in order to cover all - important we start iteration from 0 hence we should use fld ...
maxX, maxY ,maxZ- maximum possible x and y - used for bound checking
totalX,totalY,totalZ - holding the results of summation of x,y and z's
totalCount - total number of non 0 entries

"""
function meansMahalinobisKernel(goldArr,segmArr
    ,numberToLooFor
    ,loopYdim::UInt32
    ,loopXdim::UInt32
    ,loopZdim::UInt32
    ,arrDims::Tuple{UInt32,UInt32,UInt32}
    ,totalXGold,totalYGold,totalZGold,totalCountGold
    ,totalXSegm,totalYSegm,totalZSegm,totalCountSegm  )

#summing coordinates of all voxels we are intrested in 
sumX,sumY,sumZ = UInt64(0),UInt64(0),UInt64(0)
#count how many voxels of intrest there are so we will get means
count::UInt16 = UInt16(0)
#for storing results from warp reductions
shmemSum= @cuStaticSharedMem(UInt32, (32,4))   
clearSharedMemWarpLong(shmemSum, UInt8(4))
#just needed for reductions
offsetIter = UInt8(1)
#### first we analyze gold standard array
@iter3d arrDims loopXdim loopYdim  loopZdim if(  @inbounds(goldArr[x,y,z])  ==numberToLooFor)
       #updating variables needed to calculate means
       sumX+=UInt64(x) ;  sumY+=UInt64(y)  ; sumZ+=UInt64(z)   ; count+=UInt16(1)   
 end#if bool in arr  

#tell what variables are to be reduced and by what operation
@redWitAct(offsetIter,shmemSum,  sumX,+,     sumY,+    ,sumZ,+   ,count,+)
#now we have needed values in  shmemSum[1,1] - sumX  shmemSum[1,2] - sumY shmemSum[1,3] - sumZ and in shmemSum[1,4] - offsetIter
#important we need to send values to the atomics on the same warps as we did reduction 
#so we can avoid thread synchronization in such situation
@addAtomic(shmemSum,totalXGold, totalYGold ,totalZGold,totalCountGold)

### now analyzing segmentation array
#resetting variables
sumX,sumY,sumZ = UInt64(0),UInt64(0),UInt64(0)
count= UInt16(0)
clearSharedMemWarpLong(shmemSum, UInt8(4))
offsetIter = UInt8(1)

@iter3d arrDims loopXdim loopYdim  loopZdim if(  @inbounds(segmArr[x,y,z])  ==numberToLooFor)
    sumX+=UInt64(x) ;  sumY+=UInt64(y)  ; sumZ+=UInt64(z)   ; count+=UInt16(1)   
end#if bool in arr  
@redWitAct(offsetIter,shmemSum,  sumX,+,     sumY,+    ,sumZ,+   ,count,+)
@addAtomic(shmemSum,totalXSegm, totalYSegm ,totalZSegm,totalCountSegm)

return  
end


end#MeansMahalinobis





# """
# in order to avoid overfilling of local result list we need to from time to time push it into the global 
# and clear it
# """
# function pushlocalResToGlobal(intermidiateResX, intermidiateResY, intermidiateResZ,intermediateResCounter, resList, resListCounter,currVal )
#     #opdate the counter for the global list use old as offset where we start to put our results
#     if(currVal>1)
#         #oldLocal = CUDA.atomic_xchg!(pointer(intermediateResCounter), UInt32(1))
#         intermediateResCounter[1]= UInt32(1)
#         oldCount::UInt32 = @inbounds @atomic resListCounter[]+=UInt32(currVal-UInt32(1))

#             #pushing from local to global queue
#             @unroll for i in 1:currVal
#                 @inbounds  resList[oldCount+i,1] = intermidiateResX[i]
#                 @inbounds  resList[oldCount+i,2] = intermidiateResY[i]
#                 @inbounds  resList[oldCount+i,3] = intermidiateResZ[i]
#                 CUDA.@cuprint "intermidiateRes[i,1] $(intermidiateResX[i])  intermidiateRes[i,2] $(intermidiateResY[i]) intermidiateRes[i,3] $(intermidiateResZ[i])  \n "

#             end#for z dim    
#         #reset local counter
#     end    
# end
