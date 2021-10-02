

module MeansMahalinobis
using Main.BasicPreds, Main.CUDAGpuUtils, CUDA
"""
arrToAnalyze - array we analyze 
numberToLooFor - number we are intrested in in the array
loopYdim, loopXdim, loopZdim - number of times we nned to loop over those dimensions in order to cover all - important we start iteration from 0 hence we should use fld ...
maxX, maxY ,maxZ- maximum possible x and y - used for bound checking
resList - 3 column table with x,y,z coordinates of points in arrToAnalyze that are equal to numberToLooFor - we will populate the 
resListCounter - points to the length of the list
intermidiateResLength - the size of the intermediate  queue that will be used to store locally results before sending them in bulk to global memory
intermediateresCheck - when intermediate result counter will reach this number on a y loop iteration we will send values to the resList and clear intermediate queue in order to prevent its overflow

"""
function meansMahalinobisKernel(arrToAnalyze
                             ,numberToLooFor
                             ,loopYdim, loopXdim,loopZdim
                             ,maxX, maxY,maxZ
                             ,resList
                             ,resListCounter
                             ,intermidiateResLength::UInt16
                             ,intermediateresCheck::UInt16
                            )
#offset for lloking for values in source arrays 

#summing coordinates of all voxels we are intrested in 
sumX = UInt32(0)
sumY = UInt32(0)
sumZ = UInt32(0)
#count how many voxels of intrest there are so we will get means
count = UInt16(0)
#for storing results from warp reductions
shmemSum = @cuStaticSharedMem(UInt32, (33,4))   
#stroing intermediate results  that will be later send in bulk to the resList
intermidiateRes =@cuDynamicSharedMem(UInt32, (33,4)) 
#will point where we can add locally next result and will also give clue when we should reset it and send to global res array
intermediateResCounter = @cuStaticSharedMem(UInt16,1) 
#reset shared memory
@ifY 1 shmemSum[threadIdxX(),1]=0 ;   @ifY 2 shmemSum[threadIdxX(),2]=0;   @ifY 3 shmemSum[threadIdxX(),2]=0;   @ifY 4 shmemSum[threadIdxX(),2]=0
sync_threads()

#iterating over in a loop
@unroll for zdim in 0:loopZdim
    z= zdim+ blockIdxX() 
    if(z<= maxZ)      
        offset = (xWidth*yWidth *(loopZdim))    
        @unroll for ydim in 0:loopYdim# k is effectively y dimension
            y = (ydim* blockDimY()) +threadIdxY()
            if(y<=maxY)
                @unroll for xdim in 0:loopXdim
                    x=(xdim* blockDimX()) +threadIdxX()
                    if(x <=maxX)
                        if(arrToAnalyze[threadIdxX()+kx*xdim, k, blockIdx().x]==numberToLooFor)
                            # updating variables needed to calculate means
                            sumX+=x  ;  sumY+=y  ; sumZ+=z   ; count+=1 
                            #updating local quueue and counter
                            old = @inbounds @atomic intermediateResCounter+=UInt16(1)
                            intermidiateRes[old,:]=[x,y,z]
                            #CUDA.@cuprint " threadIdxX() $(threadIdxX())   kx $(kx)   xdim $(xdim)   threadIdxX()+kx*xdim  $(threadIdxX()+kx*xdim) \n"    
                        end#if bool in arr  
                    end#if xdim ok 
                end#for x dim 
            end#if y dim ok
            #here is the point where we check is the local queueis not filled too much and if so we transfer its elements to the global memory
            sync_warp()#to reduce thread divergence
            #check is it time to push to global res list
            if(intermediateResCounter[1]<intermediateresCheck)

            end#if tie to push to global res list    
        end#for  yLoops 
    end#if z dim ok 
end#for z dim





@unroll for k in 1:loopNumb
    if(threadIdxX()+(threadIdxY()-1)*32+k*1024 <=pixelNumberPerBlock)
       ind =offset+ threadIdxX()+(threadIdxY()-1)*32+k*1024
       if()

       end 
       boolGold = goldBoolGPU[ind]==numberToLooFor  
       boolSegm = segmBoolGPU[ind]==numberToLooFor     
         @inbounds locArr[ (boolGold & boolSegm + boolSegm +1) ]+=(boolGold | boolSegm)
    end#if 
end#for

offsetIter = UInt16(1)
while(offsetIter <32) 
  @inbounds locArr[3]+=shfl_down_sync(FULL_MASK, locArr[3], offsetIter)  
  @inbounds locArr[2]+=shfl_down_sync(FULL_MASK, locArr[2], offsetIter)  
  @inbounds locArr[1]+=shfl_down_sync(FULL_MASK, locArr[1], offsetIter)  
  offsetIter<<= 1
end
#shmemSum[threadIdxX(),3]+=locArr[3]
if(threadIdxX()==1)
  @inbounds shmemSum[threadIdxY(),3]+=locArr[3]
  @inbounds shmemSum[threadIdxY(),2]+=locArr[2]
  @inbounds shmemSum[threadIdxY(),1]+=locArr[1]
end

sync_threads()
if(threadIdxY()==1)
  offsetIter = UInt16(1)
  while(offsetIter <32) 
    @inbounds shmemSum[threadIdxX(),3]+=shfl_down_sync(FULL_MASK, shmemSum[threadIdxX(),3], offsetIter)  
    @inbounds shmemSum[threadIdxX(),2]+=shfl_down_sync(FULL_MASK, shmemSum[threadIdxX(),2], offsetIter)  
    @inbounds shmemSum[threadIdxX(),1]+=shfl_down_sync(FULL_MASK, shmemSum[threadIdxX(),1], offsetIter)  
    offsetIter<<= 1
  end
end  
#now we have needed values in  shmemSum[1,2] shmemSum[1,3] and shmemSum[1,1]
sync_threads()
#no point in calculating anything if we have 0 
if((shmemSum[1,3] + shmemSum[1,2] +shmemSum[1,1]) >0)
@ifXY 1 1  @inbounds @atomic tp[]+= shmemSum[1,3]
@ifXY 1 2  @inbounds @atomic fp[]+= shmemSum[1,2]
@ifXY 1 3  @inbounds @atomic fn[]+= shmemSum[1,1]
#calculated if we are intrewested in given slice wise metrics
if(conf.sliceWiseMatrics)
@ifXY 1 4  @inbounds sliceMetricsTupl[1][blockIdxX()]=shmemSum[1,3]
@ifXY 1 5  @inbounds sliceMetricsTupl[2][blockIdxX()]=shmemSum[1,2]
@ifXY 1 6  @inbounds sliceMetricsTupl[3][blockIdxX()]=shmemSum[1,1]

getMetrics(shmemSum[1,3], shmemSum[1,2], shmemSum[1,1] , pixelNumberPerSlice-(shmemSum[1,3] + shmemSum[1,2] +shmemSum[1,1]) ,sliceMetricsTupl,conf ,blockIdxX())

end#if
end#if

return  
end

"""
in order to avoid overfilling of local result list we need to from time to time push it into the global 
and clear it
"""
function pushlocalResToGlobal(intermidiateRes,intermediateResCounter, resList, resListCounter )
    
end





end#MeansMahalinobis