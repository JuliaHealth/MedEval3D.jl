

module MeansMahalinobis
using Main.BasicPreds, Main.CUDAGpuUtils, CUDA
"""
IMPORTANT x dim of threadblock needs to be always 32 
arrToAnalyze - array we analyze 
numberToLooFor - number we are intrested in in the array
loopYdim, loopXdim, loopZdim - number of times we nned to loop over those dimensions in order to cover all - important we start iteration from 0 hence we should use fld ...
maxX, maxY ,maxZ- maximum possible x and y - used for bound checking
resList - 3 column table with x,y,z coordinates of points in arrToAnalyze that are equal to numberToLooFor - we will populate the 
resListCounter - points to the length of the list
intermidiateResLength - the size of the intermediate  queue that will be used to store locally results before sending them in bulk to global memory
intermediateresCheck - when intermediate result counter will reach this number on a y loop iteration we will send values to the resList and clear intermediate queue in order to prevent its overflow
warpNumber - number of warps - as x dim of thread block needs to be 32 warp number = y dim of thread block
totalX,totalY,totalZ - holding the results of summation of x,y and z's
totalCount - total number of non 0 entries
blocks - number of thread blocks launched
"""
function meansMahalinobisKernel(arrToAnalyze
                             ,numberToLooFor
                             ,loopYdim::UInt32
                             ,loopXdim::UInt32
                             ,loopZdim::UInt32
                             ,maxX::UInt32, maxY::UInt32,maxZ::UInt32
                             ,resList
                             ,resListCounter
                             ,intermediateresCheck
                             ,totalX,totalY,totalZ
                             ,totalCount
                             ,blocks::UInt32
                             ,debugArr
                             ,dynamicMemoryLength
                            )   
    #warp number and lane number 
    wid::UInt8, lane::UInt8 = fldmod1( ((threadIdxY() -1)*blockDimY()+ threadIdxX()), warpsize())
    #summing coordinates of all voxels we are intrested in 
    sumX::UInt64 = UInt64(0)
    sumY::UInt64 = UInt64(0)
    sumZ::UInt64 = UInt64(0)
    #count how many voxels of intrest there are so we will get means
    count::UInt16 = UInt16(0)
    #for storing results from warp reductions
    shmemSum= @cuStaticSharedMem(UInt32, (33,4))   
    #stroing intermediate results  that will be later send in bulk to the resList
    intermidiateResX = @cuDynamicSharedMem(UInt16, (dynamicMemoryLength)) 
    intermidiateResY = @cuDynamicSharedMem(UInt16, (dynamicMemoryLength)) 
    intermidiateResZ = @cuDynamicSharedMem(UInt16, (dynamicMemoryLength)) 


    #will point where we can add locally next result and will also give clue when we should reset it and send to global res array
    intermediateResCounter = @cuStaticSharedMem(UInt32,(1)) 
    @widL 1 1 intermediateResCounter[1]=UInt32(1)
    #reset shared memory
    @unroll for i::UInt8 in UInt8(1):UInt8(4)
        @wid i shmemSum[threadIdxX(),i]=UInt32(0)
    end#for 
    sync_threads()
    #iterating over in a loop
    @unroll for zdim::UInt32 in UInt32(0):UInt32(loopZdim)
        z::UInt32 = (zdim*blocks) + blockIdxX()#multiply by blocks to avoid situation that a single block of threads would have no work to do
            @unroll for ydim::UInt32 in UInt32(0):UInt32(loopYdim)
                y::UInt32 = ((ydim* blockDimY()) +threadIdxY())
                if((y<=maxY) && (z<= maxZ) )
                    @unroll for xdim::UInt32 in UInt32(0):UInt32(loopXdim)
                        x::UInt32=(xdim* 32) +threadIdxX()
                        if(x <=maxX)
                            if(  @inbounds(arrToAnalyze[x,y,z])  ==numberToLooFor)
                                #updating variables needed to calculate means
                                sumX+=UInt64(x) ;  sumY+=UInt64(y)  ; sumZ+=UInt64(z)   ; count+=UInt16(1) 
                                #updating local quueue and counter
                                old::UInt32 = @inbounds @atomic intermediateResCounter[1]+=UInt32(1) #CUDA.atomic_inc!(pointer(intermediateResCounter), UInt32(1))
                                @inbounds  intermidiateResX[old]=x
                                @inbounds  intermidiateResY[old]=y
                                @inbounds  intermidiateResZ[old]=z
                            end#if bool in arr  
                        end#if xdim ok 
                    end#for x dim 
                end#if y and z dim ok
                #here is the point where we check is the local queueis not filled too much and if so we transfer its elements to the global memory
                sync_threads()#to reduce thread divergence
                #check is it time to push to global res list
                # if(intermediateResCounter[1]<intermediateresCheck)
                #     pushlocalResToGlobal(intermidiateRes,intermediateResCounter, resList, resListCounter )
                # end#if time to push to global res list    
            end#for  yLoops 
    end#for z dim

    # #push all that is left
    # pushlocalResToGlobal(intermidiateRes,intermediateResCounter, resList, resListCounter )
    # sync_threads()



    # offsetIter = UInt16(1)
    # while(offsetIter <32) 
    #     @inbounds sumX+=shfl_down_sync(FULL_MASK, sumX, offsetIter)  
    #     @inbounds sumY+=shfl_down_sync(FULL_MASK, sumY, offsetIter)  
    #     @inbounds sumZ+=shfl_down_sync(FULL_MASK, sumZ, offsetIter)  
    #     @inbounds count+=shfl_down_sync(FULL_MASK, count, offsetIter)  
    # offsetIter<<= 1
    # end
    # if(lane==1)
    #     @inbounds shmemSum[wid,1]+=sumX
    #     @inbounds shmemSum[wid,2]+=sumY
    #     @inbounds shmemSum[wid,3]+=sumZ
    #     @inbounds shmemSum[wid,4]+=count
    # end

    # sync_threads()
    # offsetIter = UInt16(1)
    # #final reduction
    # @unroll for i in 1:4
    #     finalReduction(i,offsetIter,shmemSum)
    # end#for 
    # #now we have needed values in  shmemSum[1,1] - sumX  shmemSum[1,2] - sumY shmemSum[1,3] - sumZ and in shmemSum[1,4] - count
    # sync_threads()

    # #no point in calculating anything if we have 0 
    # @widL 1 1 if(shmemSum[1,1]>0)   @inbounds @atomic totalX[]+= shmemSum[1,1] end
    # @widL 2 2 if(shmemSum[1,2]>0)   @inbounds @atomic totalY[]+= shmemSum[1,2] end
    # @widL 3 3 if(shmemSum[1,3]>0)   @inbounds @atomic totalZ[]+= shmemSum[1,3] end
    # @widL 4 4 if(shmemSum[1,4]>0)   @inbounds @atomic totalCount[]+= shmemSum[1,4] end

return  
end

"""
in order to avoid overfilling of local result list we need to from time to time push it into the global 
and clear it
"""
function pushlocalResToGlobal(intermidiateRes,intermediateResCounter, resList, resListCounter )
    #opdate the counter for the global list use old as offset where we start to put our results
    oldCount = @inbounds @atomic resListCounter[]+=UInt16(intermediateResCounter[1])
        #pushing from local to global queue
        @unroll for i in 1:intermediateResCounter[1]
            @inbounds  resList[oldCount+i,:] = @inbounds  intermidiateRes[i]
        end#for z dim    
    #reset local counter
    @inbounds intermediateResCounter[1]=0
end


"""
coordinates final round of reductions in shared memory
"""
function finalReduction(numb,offsetIter,shmemSum)
    if(threadIdxY()==numb)
        while(offsetIter <32) 
          @inbounds shmemSum[threadIdxX(),numb]+=shfl_down_sync(FULL_MASK, shmemSum[threadIdxX(),numb], offsetIter)  
          offsetIter<<= 1
        end
      end  

end




end#MeansMahalinobis