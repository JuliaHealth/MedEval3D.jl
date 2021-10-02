

module MeansMahalinobis
using Main.BasicPreds, Main.CUDAGpuUtils, CUDA

function meansMahalinobisKernel(arrToAnalyze
    ,sliceMetricsTupl
    ,tp,tn,fp,fn#tp,tn,fp,fn
    ,loopNumb#loopNumb
    ,pixelNumberPerBlock#pixelNumberPerSlice
    ,numberToLooFor#numberToLooFor
    ,conf)
#offset for lloking for values in source arrays 
    offset = (pixelNumberPerBlock*(blockIdx().x-1))

#creates shared memory and initializes it to 0
shmemSum = createAndInitializeShmem(threadIdxX(),threadIdxY())
sync_threads()
# incrementing appropriate number of times 
anyPositive = false # true If any bit will bge positive in this array - we are not afraid of data race as we can set it multiple time to true

#summing coordinates of all voxels we are intrested in 
sumX = UInt32(0)
sumY = UInt32(0)
sumZ = UInt32(0)
#count how many voxels of intrest there are so we will get means
count = UInt16(0)
 #for storing results from warp reductions
 shmemSum = @cuStaticSharedMem(UInt32, (33,4))   #thread local values that are meant to store some results - like means ... 
 #reset shared memory
 @ifY 1 shmemSum[threadIdxX(),1]=0 ;   @ifY 2 shmemSum[threadIdxX(),2]=0;   @ifY 3 shmemSum[threadIdxX(),2]=0;   @ifY 4 shmemSum[threadIdxX(),2]=0
sync_threads()

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

end#MeansMahalinobis