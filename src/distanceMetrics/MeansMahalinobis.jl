

module MeansMahalinobis
using Main.BasicPreds, Main.CUDAGpuUtils, CUDA, Main.IterationUtils, Main.ReductionUtils, Main.MemoryUtils
export meansMahalinobisKernel


"""
part of the above kernel that will be executed in order to get means
first of the gold standard array and then for other
    arrAnalyzed- gold standard or other array
    countPerZ- global memory array holding per slice counts of trues
    
"""
macro iterateForMeans(countPerZ,arrAnalyzed)
 return esc(quote

    @iter3dAdditionalzActs(arrDims,loopXdim,loopYdim,loopZdim,
    #inner expression
    if(  @inbounds($arrAnalyzed[x,y,z])  ==numberToLooFor)
        #updating variables needed to calculate means
        sumX+=Float32(x) ;  sumY+=Float32(y)  ; sumZ+=Float32(z)   ; count+=Float32(1)   
    end,
    #after z expression - we get slice wise true counts from it 
    begin
        sync_threads()
        #reducing count only
        if(z<=arrDims[3])
            countTemp = count
            @redWitAct(offsetIter,shmemSum, count,+)
            #saving to global memory count of this slice
            @ifXY 1 1 begin 
                 $countPerZ[z]=(shmemSum[1,1] - oldZVal[1] )
                oldZVal[1]=shmemSum[1,1]
            end
            #clear shared memory only first row was used and sync threads 
            clearSharedMemWarpLong(shmemSum, UInt8(1), Float32(0.0))
            count=countTemp#to preserve proper value for total count
        end#if ar dims
    end )#if bool in arr  

    #tell what variables are to be reduced and by what operation
    @redWitAct(offsetIter,shmemSum,  sumX,+,     sumY,+    ,sumZ,+   ,count,+)
     
end)
end



"""
IMPORTANT x dim of threadblock needs to be always 32 
goldArr,segmArr  - arrays we analyze 
numberToLooFor - number we are intrested in in the array
loopYdim, loopXdim, loopZdim - number of times we nned to loop over those dimensions in order to cover all - important we start iteration from 0 hence we should use fld ...
maxX, maxY ,maxZ- maximum possible x and y - used for bound checking
totalX,totalY,totalZ - holding the results of summation of x,y and z's
totalCount - total number of non 0 entries
countPerZ - count of non empty entries per slice - particularly usefull in case it is 0
covariancesSliceWise - slice wise covariances - needed in case we want slice wise results - a matrix where each row entry is 
        I)gold standard mask values
        1)variance x    2)cov xy     3)cov xz     4)var y    5)cov yz      6)var z 
        II) other mask 
        7)variance x    8)cov xy     9)cov xz     10)var y    11)cov yz      12)var z         
covarianceGlobal (just one column but entries exactly the same as above)
mahalanobisResGlobal - global result of Mahalinobis distance
mahalanobisResSliceWise - global result of Mahalinobis distance
"""
function meansMahalinobisKernel(goldArr,segmArr
    ,numberToLooFor
    ,loopYdim::UInt32
    ,loopXdim::UInt32
    ,loopZdim::UInt32
    ,arrDims::Tuple{UInt32,UInt32,UInt32}
    ,totalXGold,totalYGold,totalZGold,totalCountGold
    ,totalXSegm,totalYSegm,totalZSegm,totalCountSegm
    ,countPerZGold,countPerZSegm,covariancesSliceWise,covarianceGlobal,mahalanobisResGlobal, mahalanobisResSliceWise   )

    grid_handle = this_grid()
    # keeping counter of old z value - in order to be able to get slicewise z counter
    oldZVal= @cuStaticSharedMem(Float32, (1))
    #summing coordinates of all voxels we are intrested in 
    sumX,sumY,sumZ = Float32(0),Float32(0),Float32(0)
    #count how many voxels of intrest there are so we will get means
    count = Float32(0)
    #for storing results from warp reductions
    shmemSum= @cuStaticSharedMem(Float32, (32,6))   
    clearSharedMemWarpLong(shmemSum, UInt8(6), Float32(0.0))
    @ifXY 1 1 oldZVal[1]=0
    #just needed for reductions
    offsetIter = UInt8(1)
    #### first we analyze gold standard array
    sync_threads()

    @iterateForMeans(countPerZGold,goldArr)
  

    #now we have needed values in  shmemSum[1,1] - sumX  shmemSum[1,2] - sumY shmemSum[1,3] - sumZ and in shmemSum[1,4] - offsetIter
    #important we need to send values to the atomics on the same warps as we did reduction 
    #so we can avoid thread synchronization in such situation
    @addAtomic(shmemSum,totalXGold, totalYGold ,totalZGold,totalCountGold)
    
    sync_threads()
    ### now analyzing segmentation array
    #resetting variables
    sumX,sumY,sumZ = Float32(0),Float32(0),Float32(0)
    count= Float32(0)
    clearSharedMemWarpLong(shmemSum, UInt8(6), Float32(0.0))
    @ifXY 1 1 oldZVal[1]=0
    sync_threads()
    #iterations
    @iterateForMeans(countPerZSegm,segmArr)

    @addAtomic(shmemSum,totalXSegm, totalYSegm ,totalZSegm,totalCountSegm)

    sync_grid(grid_handle)
##################### getting covariances
  
##first gold mask

    #at this spot we should have all counts of x,y,z and counts - we can get means from it
    clearSharedMemWarpLong(shmemSum, UInt8(6), Float32(0.0))
    # we are using shared memory to hold means of 1)x 2)y and 3)z 
    meanxyz= @cuStaticSharedMem(Float64, (3))
    #for storing intermediate results in shared memory
    #1)variance x    2)cov xy     3)cov xz     4)var y    5)cov yz      6)var z 
    intermedieteResVarX= @cuStaticSharedMem(Float64, (1))
    intermedieteResCovxy= @cuStaticSharedMem(Float64, (1))
    intermedieteResCovxz= @cuStaticSharedMem(Float64, (1))
    intermedieteResVarY= @cuStaticSharedMem(Float64, (1))
    intermedieteResCovyz= @cuStaticSharedMem(Float64, (1))
    intermedieteResVarZ= @cuStaticSharedMem(Float64, (1))


    @ifXY 1 1  meanxyz[1]= totalXGold[1]/totalCountGold[1]
    @ifXY 1 2  meanxyz[2]= totalYGold[1]/totalCountGold[1]
    @ifXY 1 3  meanxyz[3]= totalZGold[1]/totalCountGold[1]
    #from now one sumX is variance of x ,sumY is covariance xy ,sumZ is covariance xz 
    sumX,sumY,sumZ ,count= Float32(0.0),Float32(0.0),Float32(0.0),Float32(0.0)
    sync_threads()

    @iter3dAdditionalxyzActsAndZcheck(arrDims,loopXdim,loopYdim,loopZdim
    #z check
    ,(z<= arrDims[3] && countPerZGold[z]>0.0),
    #inner expression
    if(  @inbounds(goldArr[x,y,z])  ==numberToLooFor)
        #getting variance x and x covariances
        sumX+=(Float32(x)-meanxyz[1])^2
        sumY+=((Float32(y)-meanxyz[2])*(Float32(x)-meanxyz[1]))
        sumZ+=((Float32(z)-meanxyz[3])*(Float32(x)-meanxyz[1]))
        count+=Float32(1)   
    end,
    #x additional fun - currently nothing to implement here
    begin    end,
    #y additional fun - as we added all x variances and covariances we need now to update y variance and y covariance 
    #we need to add this the same amount of time as we added the x variables so we use count for it
    # of course if count is 0 we can ignore this step
    begin
   #we put reduced values into share memory 
   @redOnlyStepOne(offsetIter,shmemSum,  sumX,+,     sumY,+    ,sumZ,+   ,count,+)
   if(threadIdxX()==1 && count>0)
       @inbounds shmemSum[threadIdxY(),1]+= sumX
       @inbounds shmemSum[threadIdxY(),2]+= sumY
       @inbounds shmemSum[threadIdxY(),3]+= sumZ
       @inbounds shmemSum[threadIdxY(),4]+= count
       #putting  variance y and covariance yz manually to shared memory multiply appropriate amount of time
       @inbounds shmemSum[threadIdxY(),5]+= count*(y-meanxyz[2])^2#variance y
       @inbounds shmemSum[threadIdxY(),6]+= count*((y-meanxyz[2])*(z-meanxyz[3]))#covariance yz
   end;
   #reset as values are already saved in shmemsum
   sumX,sumY,sumZ ,count= Float32(0.0),Float32(0.0),Float32(0.0),Float32(0.0)
   sync_warp()

    end,
    #z additional fun
    begin
        sync_threads()
        #we do the last step of reductions to get all of the values into first spots of shared memory
        @redOnlyStepThree(offsetIter,shmemSum, +,+,+  ,+,+,+)
        #we will use it later to get slicewise results and in the end we will send those to global memory
       
        shmemSum, intermedieteResVarX,intermedieteResCovxy,intermedieteResCovxz,intermedieteResVarY,intermedieteResCovyz,intermedieteResVarZ
                
        #now we need to add slicewise results 
        covariancesSliceWise[shmemSum[1,1] ]
     
     
        #clear shared memory
        clearSharedMemWarpLong(shmemSum, UInt8(6), Float32(0.0))

    end  )# iterations loop

    end

    return  
    end#meansMahalinobisKernel







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



# function meansMahalinobisKernel(goldArr,segmArr
#     ,numberToLooFor
#     ,loopYdim::UInt32
#     ,loopXdim::UInt32
#     ,loopZdim::UInt32
#     ,arrDims::Tuple{UInt32,UInt32,UInt32}
#     ,totalXGold,totalYGold,totalZGold,totalCountGold
#     ,totalXSegm,totalYSegm,totalZSegm,totalCountSegm  )

#     #summing coordinates of all voxels we are intrested in 
#     sumX,sumY,sumZ = UInt64(0),UInt64(0),UInt64(0)
#     #count how many voxels of intrest there are so we will get means
#     count::UInt16 = UInt16(0)
#     #for storing results from warp reductions
#     shmemSum= @cuStaticSharedMem(UInt32, (32,4))   
#     clearSharedMemWarpLong(shmemSum, UInt8(4))
#     #just needed for reductions
#     offsetIter = UInt8(1)
#     #### first we analyze gold standard array
#     @iter3d arrDims loopXdim loopYdim  loopZdim if(  @inbounds(goldArr[x,y,z])  ==numberToLooFor)
#         #updating variables needed to calculate means
#         sumX+=UInt64(x) ;  sumY+=UInt64(y)  ; sumZ+=UInt64(z)   ; count+=UInt16(1)   
#     end#if bool in arr  

#     #tell what variables are to be reduced and by what operation
#     @redWitAct(offsetIter,shmemSum,  sumX,+,     sumY,+    ,sumZ,+   ,count,+)
#     #now we have needed values in  shmemSum[1,1] - sumX  shmemSum[1,2] - sumY shmemSum[1,3] - sumZ and in shmemSum[1,4] - offsetIter
#     #important we need to send values to the atomics on the same warps as we did reduction 
#     #so we can avoid thread synchronization in such situation
#     @addAtomic(shmemSum,totalXGold, totalYGold ,totalZGold,totalCountGold)

#     ### now analyzing segmentation array
#     #resetting variables
#     sumX,sumY,sumZ = UInt64(0),UInt64(0),UInt64(0)
#     count= UInt16(0)
#     clearSharedMemWarpLong(shmemSum, UInt8(4))
#     offsetIter = UInt8(1)

#     @iter3d arrDims loopXdim loopYdim  loopZdim if(  @inbounds(segmArr[x,y,z])  ==numberToLooFor)
#         sumX+=UInt64(x) ;  sumY+=UInt64(y)  ; sumZ+=UInt64(z)   ; count+=UInt16(1)   
#     end#if bool in arr  
#     @redWitAct(offsetIter,shmemSum,  sumX,+,     sumY,+    ,sumZ,+   ,count,+)
#     @addAtomic(shmemSum,totalXSegm, totalYSegm ,totalZSegm,totalCountSegm)

#     return  
#     end#meansMahalinobisKernel

# end#MeansMahalinobis
