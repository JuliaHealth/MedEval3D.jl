"""
holding kernel and necessery functions to calclulate number of true positives,
true negatives, false positives and negatives par image and per slice
using synergism described by Taha et al. this will enable later fast calculations of many other metrics
"""
module TpfpfnKernel
export getTpfpfnData

using CUDA, Main.GPUutils, Logging


"""
returning the data  from a kernel that  calclulate number of true positives,
true negatives, false positives and negatives par image and per slice in given data 
goldBoolGPU - array holding data of gold standard bollean array
segmBoolGPU - boolean array with the data we want to compare with gold standard
tp,tn,fp,fn - holding single values for true positive, true negative, false positive and false negative
intermediateResTp, intermediateResFp, intermediateResFn - arrays holding slice wise results for true positive ...
threadNumPerBlock = threadNumber per block defoult is 512
numberToLooFor - num
IMPORTANT - in the ned of the goldBoolGPU and segmBoolGPU one need  to add some  additional number of 0=falses - number needs to be the same as indexCorr
IMPORTANT - currently block sizes of 512 are supported only
"""
function getTpfpfnData!(goldBoolGPU
    , segmBoolGPU
    ,tp,tn,fp,fn
    ,intermediateResTp
    ,intermediateResFp
    ,intermediateResFn
    ,pixelNumberPerSlice::Int64
    ,numberOfSlices::Int64
    ,numberToLooFor::T
    ,IndexesArray
    ,threadNumPerBlock::Int64 = 512) where T

loopNumb, indexCorr = getKernelContants(threadNumPerBlock,pixelNumberPerSlice)
args = (goldBoolGPU,segmBoolGPU,tp,tn,fp,fn, intermediateResTp,intermediateResFp,intermediateResFn, loopNumb, indexCorr,Int64(round(threadNumPerBlock/32)),pixelNumberPerSlice,numberToLooFor,IndexesArray)

@info "numberOfSlices" numberOfSlices
@info "loopNumb" loopNumb
@info "indexCorr" indexCorr

@cuda threads=threadNumPerBlock blocks=numberOfSlices getBlockTpFpFn(args...) 

end#getTpfpfnData

"""
adapted from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/src/mapreduce.jl
goldBoolGPU - array holding data of gold standard bollean array
segmBoolGPU - boolean array with the data we want to compare with gold standard
tp,tn,fp,fn - holding single values for true positive, true negative, false positive and false negative
intermediateResTp, intermediateResFp, intermediateResFn - arrays holding slice wie results for true positive ...
loopNumb - number of times the single lane needs to loop in order to get all needed data
sliceEdgeLength - length of edge of the slice we need to square this number to get number of pixels in a slice
amountOfWarps - how many warps we can stick in the vlock
"""
function getBlockTpFpFn(goldBoolGPU
        , segmBoolGPU
        ,tp,tn,fp,fn
        ,intermediateResTp
        ,intermediateResFp
        ,intermediateResFn
        ,loopNumb::Int64
        ,indexCorr::Int64
        ,amountOfWarps::Int64
        ,pixelNumberPerSlice::Int64
        ,numberToLooFor::T
        ,IndexesArray) where T
    # we multiply thread id as we are covering now 2 places using one lane - hence after all lanes gone through we will cover 2 blocks - hence second multiply    
    correctedIdx = (threadIdx().x-1)* indexCorr+1
    i = correctedIdx + (pixelNumberPerSlice*(blockIdx().x-1))
    #i = correctedIdx + ((blockIdx().x - 1) *indexCorr) * (blockDim().x)# used as a basis to get data we want from global memory
   wid, lane = fldmod1(threadIdx().x,32)
#creates shared memory and initializes it to 0
   shmem,shmemSum = createAndInitializeShmem(wid,threadIdx().x,amountOfWarps,lane)
   shmem[513,1]= numberToLooFor
# incrementing appropriate number of times 

    @unroll for k in 0:loopNumb
    if(correctedIdx+k<=pixelNumberPerSlice)
        incr_shmem(threadIdx().x,goldBoolGPU[i+k]==shmem[513,1],segmBoolGPU[i+k]==shmem[513,1],shmem)
        #incr_shmem(threadIdx().x+1,goldBoolGPU[i+k]==shmem[513,1],segmBoolGPU[i+k]==shmem[513,1],shmem,IndexesArray)
    end    
    end#for
    #IndexesArray[i]= shmem[threadIdx().x,3]
            # if(goldBoolGPU[i+k]==shmem[513,1] && segmBoolGPU[i+k]==shmem[513,1])
        #     IndexesArray[i+k]=1
        # end
        #atomic tp[]+=1
        # IndexesArray[i+k]=i+k
        # #if(blockIdx().x>178 && threadIdx().x==1) @cuprint "blockIdx().x $(blockIdx().x)"    end
    
    # @unroll for k in 0:loopNumb-1
    #     if(correctedIdx+k<=pixelNumberPerSlice)    
    #         incr_shmem(threadIdx().x,goldBoolGPU[i+k]==numberToLooFor,segmBoolGPU[i+k]==numberToLooFor,shmem)
    #     end    
    # end#for 
    
    ### can loop first with loop numnber -1 ; and later loop over  las elemnts - so pixel number minus what we get in the end of last loop  - this way we avoid ifs


    #reducing across the warp
    firstReduce(shmem,shmemSum,wid,threadIdx().x,lane,IndexesArray,i)

    sync_threads()
    #now all data about of intrest should be in  shared memory so we will get all rsults from warp reduction in the shared memory 
    getSecondBlockReduce( 1,3,wid,intermediateResTp,tp,shmemSum,blockIdx().x,lane)
    getSecondBlockReduce( 2,2,wid,intermediateResFp,fp,shmemSum,blockIdx().x,lane)
    getSecondBlockReduce( 3,1,wid,intermediateResFn,fn,shmemSum,blockIdx().x,lane)

   return  
   end

"""
add value to the shared memory in the position i, x where x is 1 ,2 or 3 and is calculated as described below
boolGold & boolSegm + boolGold +1 will evaluate to 
    3 in case  of true positive
    2 in case of false positive
    1 in case of false negative
"""
@inline function incr_shmem( primi::Int64,boolGold::Bool,boolSegm::Bool,shmem )
    @inbounds shmem[ primi, (boolGold & boolSegm + boolSegm +1) ]+=(boolGold | boolSegm) 
    return true
end


"""
creates shared memory and initializes it to 0
wid - the number of the warp in the block
"""
function createAndInitializeShmem(wid, threadId,amountOfWarps,lane)
   #shared memory for  stroing intermidiate data per lane  
   shmem = @cuStaticSharedMem(UInt16, (513,3))
   #for storing results from warp reductions
   shmemSum = @cuStaticSharedMem(UInt16, (33,3))
    #setting shared memory to 0 
    shmem[threadId, 3]=0
    shmem[threadId, 2]=0
    shmem[threadId, 1]=0
    
    if(wid==1)
    shmemSum[lane,1]=0
    end
    if(wid==2)
        shmemSum[lane,2]=0
    end
    if(wid==3)
    shmemSum[lane,3]=0
    end            

return (shmem,shmemSum )

end#createAndInitializeShmem


"""
reduction across the warp and adding to appropriate spots in the  shared memory
"""
function firstReduce(shmem,shmemSum,wid,threadIdx,lane,IndexesArray,i   )
  

    @inbounds shmemSum[wid,1] = reduce_warp(shmem[threadIdx,1],32)
    @inbounds shmemSum[wid,2] = reduce_warp(shmem[threadIdx,2],32)
    @inbounds shmemSum[wid,3] = reduce_warp(shmem[threadIdx,3],32)
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
function getSecondBlockReduce(chosenWid,numb,wid, intermediateRes,singleREs,shmemSum,blockId,lane)
    if(wid==chosenWid )
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