"""
holding kernel and necessery functions to calclulate number of true positives,
true negatives, false positives and negatives par image and per slice
using synergism described by Taha et al. this will enable later fast calculations of many other metrics
"""
module TpfpfnKernel
export getTpfpfnData

using CUDA, Main.CUDAGpuUtils, Logging,StaticArrays
using Main.MainOverlap, Main.RandIndex, Main.ProbabilisticMetrics, Main.VolumeMetric, Main.InformationTheorhetic


"""
returning the data  from a kernel that  calclulate number of true positives,
true negatives, false positives and negatives par image and per slice in given data 
goldBoolGPU - array holding data of gold standard bollean array
segmBoolGPU - boolean array with the data we want to compare with gold standard
tp,tn,fp,fn - holding single values for true positive, true negative, false positive and false negative
sliceMetricsTupl - tuple holding slicewiseMetrics
metricsTuplGlobal - tuple holding required metrics of all image 
threadNumPerBlock = threadNumber per block defoult is 512
numberToLooFor - num
conf- adapted ConfigurtationStruct - used to pass information what metrics should be

"""
function getTpfpfnData!(goldBoolGPU
    , segmBoolGPU
    ,tp,tn,fp,fn
    ,sliceMetricsTupl
    ,metricsTuplGlobal
    ,pixelNumberPerSlice::Int64
    ,numberOfSlices::Int64
    ,numberToLooFor::T
    ,conf
    ,totalNumberOfVoxels::Int64) where T

args = ( goldBoolGPU
        ,segmBoolGPU
        ,sliceMetricsTupl
        ,tp,tn,fp,fn
        ,cld(pixelNumberPerSlice,1024)
        ,pixelNumberPerSlice
        ,numberToLooFor
        ,conf )
#getMaxBlocksPerMultiproc(args, getBlockTpFpFn) -- evaluates to 3

#get tp,fp,fna and slicewise results if required
@cuda threads=(32,32) blocks=numberOfSlices getBlockTpFpFn(args...) 
#get global metrics tp,fp, fn,totalNumbOfVoxels,metricsTuplGlobal,conf
@cuda threads=(32,32) blocks=1  getGlobalMetricsKernel(tp,fp, fn,totalNumberOfVoxels,metricsTuplGlobal,conf)

return args
end#getTpfpfnData

"""
adapted from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/src/mapreduce.jl
goldBoolGPU - array holding data of gold standard bollean array
segmBoolGPU - boolean array with the data we want to compare with gold standard
tp,tn,fp,fn - holding single values for true positive, true negative, false positive and false negative
sliceMetricsTupl - tuple of arrays holding slice wise results for tp,fp,fn and all metrics of intrest - in case we would not be intrested in some metric tuple in this spot will have the array of length 1
loopNumb - number of times the single lane needs to loop in order to get all needed data
sliceEdgeLength - length of edge of the slice we need to square this number to get number of pixels in a slice
conf- adapted ConfigurtationStruct - used to pass information what metrics should be
"""
function getBlockTpFpFn(goldBoolGPU#goldBoolGPU
        , segmBoolGPU#segmBoolGPU
        ,sliceMetricsTupl
        ,tp,tn,fp,fn#tp,tn,fp,fn
        ,loopNumb#loopNumb
        ,pixelNumberPerSlice#pixelNumberPerSlice
        ,numberToLooFor#numberToLooFor
        ,conf)
    # we multiply thread id as we are covering now 2 places using one lane - hence after all lanes gone through we will cover 2 blocks - hence second multiply    
    offset = (pixelNumberPerSlice*(blockIdx().x-1))
    
#creates shared memory and initializes it to 0
   shmemSum = createAndInitializeShmem(threadIdxX(),threadIdxY())
# incrementing appropriate number of times 
    locArr= zeros(MVector{3,UInt16})

    @unroll for k in UInt16(0):loopNumb
        if(threadIdxX()+(threadIdxY()-1)*32+k*1024 <=pixelNumberPerSlice)
           ind =offset+ threadIdxX()+(threadIdxY()-1)*32+k*1024
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
this will be invoked in order to get global metrics besed on tp,fp,fn calculated in previous kernel 
tp,tn,fp - true positive, true negative, false positive
totalNumbOfVoxels - number of voxels in all image
metricsTuplGlobal - tuple with array of length one for storing global metrics
conf - ConfigurtationStruct - marking in what metrics we are intrested in 
"""
function getGlobalMetricsKernel(tp,fp, fn,totalNumbOfVoxels::Int64,metricsTuplGlobal,conf)
  getMetrics(tp[1],fp[1], fn[1],(totalNumbOfVoxels-(tp[1] +fn[1]+ fp[1] )) ,metricsTuplGlobal,conf,1 )
  return
end



"""
loading data into results
sliceMetricsTupl
   1) true positives
   2) false positives
   3) flse negatives
   4) dice
   5) jaccard
   6) gce
   7) randInd
   8) cohen kappa
   9) volume metric
   10) mutual information
   11) variation of information
positionToUpdate - index at which we want to update the metric - in case of slice wise metrics it will be number of slice = block idx

"""
function getMetrics(tp,fp, fn,tn,sliceMetricsTupl,conf,positionToUpdate   )
@ifXY 1 7 if (conf.dice ) @inbounds sliceMetricsTupl[4][positionToUpdate]=   MainOverlap.dice(tp,fp, fn) end 
@ifXY 1 8  if (conf.jaccard ) @inbounds sliceMetricsTupl[5][positionToUpdate]= MainOverlap.jaccard(tp,fp, fn) end 
@ifXY 1 9  if (conf.gce ) @inbounds sliceMetricsTupl[6][positionToUpdate]= MainOverlap.gce(tn,tp,fp, fn) end 
@ifXY 1 10 if (conf.randInd ) @inbounds sliceMetricsTupl[7][positionToUpdate]=  RandIndex.calculateAdjustedRandIndex(tn,tp,fp, fn) end 
@ifXY 1 11 if (conf.kc ) @inbounds sliceMetricsTupl[8][positionToUpdate]=  ProbabilisticMetrics.calculateCohenCappa(tn,tp,fp, fn  ) end 
@ifXY 1 12  if (conf.vol ) @inbounds sliceMetricsTupl[9][positionToUpdate]= VolumeMetric.getVolumMetric(tp,fp, fn ) end 
@ifXY 1 13 if (conf.mi ) @inbounds sliceMetricsTupl[10][positionToUpdate]=   InformationTheorhetic.mutualInformationMetr(tn,tp,fp, fn) end 
@ifXY 1 14 if (conf.vi ) @inbounds sliceMetricsTupl[11][positionToUpdate]=  InformationTheorhetic.variationOfInformation(tn,tp,fp, fn) end 

end






"""
add value to the shared memory in the position i, x where x is 1 ,2 or 3 and is calculated as described below
boolGold & boolSegm + boolGold +1 will evaluate to 
    3 in case  of true positive
    2 in case of false positive
    1 in case of false negative
"""
@inline function incr_locArr(boolGold::Bool,boolSegm::Bool,locArr::MVector{3, UInt16} ,shmemSum,wid)

  @inbounds locArr[ (boolGold & boolSegm + boolSegm +1) ]+=(boolGold | boolSegm)
  
    return true
end
"""
get which warp it is in a block and which lane in warp 
"""
function getWidAndLane(threadIdx)::Tuple{UInt8, UInt8}
      return fldmod1(threadIdx,32)
end

"""
creates shared memory and initializes it to 0
wid - the number of the warp in the block
"""
function createAndInitializeShmem(wid,lane)
   #for storing results from warp reductions
   shmemSum = @cuStaticSharedMem(UInt16, (33,3))

    if(wid==1)
    shmemSum[lane,1]=0
    end
    if(wid==2)
        shmemSum[lane,2]=0
    end
    if(wid==3)
    shmemSum[lane,3]=0
    end            

return shmemSum

end#createAndInitializeShmem


"""
reduction across the warp and adding to appropriate spots in the  shared memory
"""
function firstReduce(shmemSum ,locArr)
    @inbounds shmemSum[threadIdxX(),1] = reduce_warp(locArr[1],32)
    @inbounds shmemSum[threadIdxX(),2] = reduce_warp(locArr[2],32)
    @inbounds shmemSum[threadIdxX(),3] = reduce_warp(locArr[3],32)
end#firstReduce

"""
sets the final block amount of true positives, false positives and false negatives and saves it
to the  array representing each slice, 
wid - the warp in a block we want to use
numb - number associated with constant - used to access shared memory for example
chosenWid - on which block we want to make a reduction to happen
intermediateRes - array with intermediate -  slice wise results
singleREs - the final  constant holding image witde values (usefull for example for debugging)
shmemSum - shared memory where we get the  results to be reduced now and to which we will also save the output
blockId - number related to block we are currently in 
lane - the lane in the warp
"""
function getSecondBlockReduce(chosenWid,numb,wid, intermediateRes,singleREs,shmemSum,blockId,lane,IndexesArray)
    if(wid==chosenWid )
      IndexesArray[blockId+lane]=shmemSum[lane,numb]
      shmemSum[33,numb] = reduce_warp(shmemSum[lane,numb],32 )
      #probably we do not need to sync warp as shfl dow do it for us         
      if(lane==1)
        @inbounds @atomic singleREs[]+=shmemSum[33,numb]
      end    
      if(lane==2)

        @inbounds intermediateRes[blockId]=shmemSum[33,numb]
      end    
    #   if(lane==3)
    #     #ovewriting the value 
    #     @inbounds shmemSum[1,numb]=vall
    #   end     

  end  

end#getSecondBlockReduce







end#TpfpfnKernel



########### version with cooperative groups

# function getBlockTpFpFn(goldBoolGPU
#     , segmBoolGPU
#     ,tp,tn,fp,fn
#     ,intermediateResTp
#     ,intermediateResFp
#     ,intermediateResFn
#     ,loopNumb::Int64
#     ,indexCorr::Int64
#     ,amountOfWarps::Int64
#     ,pixelNumberPerSlice::Int64
#     ,numberToLooFor::T
#     ,IndexesArray
#     ,maxSlicesPerBlock::Int64
#     ,slicesPerBlockMatrix
#     ,numberOfBlocks::Int64) where T
# # we multiply thread id as we are covering now 2 places using one lane - hence after all lanes gone through we will cover 2 blocks - hence second multiply    
# correctedIdx = (threadIdxX()-1)* indexCorr+1
# i= correctedIdx
# #i = correctedIdx + ((blockIdx().x - 1) *indexCorr) * (blockDimX())# used as a basis to get data we want from global memory
# wid, lane = fldmod1(threadIdxX(),32)
# #creates shared memory and initializes it to 0
# shmem,shmemSum = createAndInitializeShmem()
# shmem[513,1]= numberToLooFor
# ##### in this outer loop we are iterating over all slices that this block is responsible for
# @unroll for blockRef in 1:maxSlicesPerBlock    
#     sliceNumb= slicesPerBlockMatrix[blockIdx().x,blockRef]
#         if(sliceNumb>0)
#             i = correctedIdx + (pixelNumberPerSlice*(sliceNumb-1))# used as a basis to get data we want from global memory
#             setShmemTo0(wid,threadIdxX(),lane,shmem,shmemSum)           
#             # incrementing appropriate number of times 
        
#         @unroll for k in 0:loopNumb
#                 if(correctedIdx+k<=pixelNumberPerSlice)
#                     incr_shmem(threadIdxX(),goldBoolGPU[i+k]==shmem[513,1],segmBoolGPU[i+k]==shmem[513,1],shmem)
#                 end#if
#             end#for   
#         #reducing across the warp
#         firstReduce(shmem,shmemSum,wid,threadIdxX(),lane,IndexesArray,i)
        
        
#         sync_threads()
#         #now all data about of intrest should be in  shared memory so we will get all rsults from warp reduction in the shared memory 
#         getSecondBlockReduce( 1,3,wid,intermediateResTp,tp,shmemSum,blockIdx().x,lane)
#         getSecondBlockReduce( 2,2,wid,intermediateResFp,fp,shmemSum,blockIdx().x,lane)
#         getSecondBlockReduce( 3,1,wid,intermediateResFn,fn,shmemSum,blockIdx().x,lane)
#     end#if     
# end#for

# return  
# end
