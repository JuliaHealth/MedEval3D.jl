using Revise, Parameters, Logging
using CUDA
include("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
include("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\gpuUtils.jl")
include("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")

include("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernels\\TpfpfnKernel.jl")
include("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\overLap\\MainOverlap.jl")
using Main.BasicPreds, Main.GPUutils,Cthulhu,BenchmarkTools , CUDA
using Main.MainOverlap, Main.TpfpfnKernel

goldBoolGPU,segmBoolGPU,tp,tn,fp,fn, tpArr,tnArr,fpArr, fnArr, blockNum , nx,ny,nz ,tpTotalTrue,tnTotalTrue,fpTotalTrue, fnTotalTrue ,tpPerSliceTrue,  tnPerSliceTrue,fpPerSliceTrue,fnPerSliceTrue ,flattG, flattSeg ,FlattGoldGPU,FlattSegGPU,intermediateResTp,intermediateResFp,intermediateResFn = getSmallTestBools();


FlattGoldGPU
FlattSegGPU

z = mean(FlattGoldGPU)
typeof(z)
sizz= size(goldBoolGPU)



"""
now plan is to create a matrix on the CPU that will be pushed onto the GPU
matrix will tell each block which slices it should manage in order to cover all slices
number of blocks will be calclulated in a way to fit them all in a single grid and enable full utilization of cooperative groups 
matrix will be n x m  where 
    n is number of blocks we can fit in the grid
    m is maximum number of slices managed per block - of course in most cases some blocks will not be needed in final round - in this case this will evaluate to 0 ...
    

    https://stackoverflow.com/questions/63929929/processing-shared-work-queue-using-cuda-atomic-operations-and-grid-synchronizati/63930239#63930239
    https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#grid-synchronization-cg

    """



"""
adapted from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/src/mapreduce.jl
goldBoolGPU - array holding data of gold standard bollean array
segmBoolGPU - boolean array with the data we want to compare with gold standard
ic - holding single value for global intercalss correlation
intermediateRes- array holding slice wise results for ic
loopNumb - number of times the single lane needs to loop in order to get all needed data
sliceEdgeLength - length of edge of the slice we need to square this number to get number of pixels in a slice
amountOfWarps - how many warps we can stick in the block
"""
function getBlockTpFpFn(goldBoolGPU::CuDeviceArray{Bool,1, 1}
        ,segmBoolGPU::CuDeviceArray{Bool, 1, 1}
        ,ic
        ,loopNumb::Int64
        ,indexCorr::Int64
        ,sliceEdgeLength::Int64
        ,amountOfWarps::Int64)
    # we multiply thread id as we are covering now 2 places using one lane - hence after all lanes gone through we will cover 2 blocks - hence second multiply    
    i = (threadIdx().x* indexCorr) + ((blockIdx().x - 1) *indexCorr) * (blockDim().x)# used as a basis to get data we want from global memory
   wid, lane = fldmod1(threadIdx().x,32)
#creates shared memory and initializes it to 0
   shmem,shmemSum = createAndInitializeShmem(wid,threadIdx().x,sliceEdgeLength,amountOfWarps)

# incrementing appropriate number of times 

    @unroll for k in 0:loopNumb
    incr_shmem(threadIdx().x,goldBoolGPU[i+k],segmBoolGPU[i+k],shmem,blockIdx().x )
   end#for 

    #reducing across the warp
    firstReduce(shmem,shmemSum,wid,threadIdx().x,lane)
    sync_threads()
    #now all data about of intrest should be in  shared memory so we will get all rsults from warp reduction in the shared memory 
    getSecondBlockReduce( 1,3,wid,intermediateResTp,tp,shmemSum,blockIdx().x,lane)
    getSecondBlockReduce( 2,2,wid,intermediateResFp,fp,shmemSum,blockIdx().x,lane)
    getSecondBlockReduce( 3,1,wid,intermediateResFn,fn,shmemSum,blockIdx().x,lane)
    sync_threads()

    # if(wid==4 && lane==1)   shmemSum[1,2]= sliceEdgeLength*sliceEdgeLength-                                end



    return  
   end



   """
   returning the data  from a kernel that  calclulate number of true positives,
   true negatives, false positives and negatives par image and per slice in given data 
   goldBoolGPU - array holding data of gold standard bollean array
   segmBoolGPU - boolean array with the data we want to compare with gold standard
   tp,tn,fp,fn - holding single values for true positive, true negative, false positive and false negative
   intermediateResTp, intermediateResFp, intermediateResFn - arrays holding slice wise results for true positive ...
   threadNumPerBlock = threadNumber per block defoult is 512
   IMPORTANT - in the ned of the goldBoolGPU and segmBoolGPU one need  to add some  additional number of 0=falses - number needs to be the same as indexCorr
   IMPORTANT - currently block sizes of 512 are supported only
   """
   function getTpfpfnData!(goldBoolGPU
       , segmBoolGPU
       ,tp,tn,fp,fn
       ,intermediateResTp
       ,intermediateResFp
       ,intermediateResFn
       ,sliceEdgeLength::Int64
       ,numberOfSlices::Int64
       ,threadNumPerBlock::Int64 = 512)
   
   blockNum, loopNumb, indexCorr = getKernelContants(threadNumPerBlock,numberOfSlices,sliceEdgeLength)
   args = (goldBoolGPU,segmBoolGPU,tp,tn,fp,fn, intermediateResTp,intermediateResFp,intermediateResFn, loopNumb, indexCorr,sliceEdgeLength,Int64(round(threadNumPerBlock/32)))
   @cuda threads=threadNumPerBlock blocks=blockNum getBlockTpFpFn(args...) 
   
   end#getTpfpfnData





"""
add value to the shared memory in the position i, x where x is 1 ,2 or 3 and is calculated as described below
boolGold & boolSegm + boolGold +1 will evaluate to 
    3 in case  of true positive
    2 in case of false positive
    1 in case of false negative
"""
@inline function incr_shmem( primi::Int64,boolGold::Bool,boolSegm::Bool,shmem,blockId )
    @inbounds shmem[ primi, (boolGold & boolSegm + boolSegm +1) ]+=(boolGold | boolSegm)
    return true
end


"""
creates shared memory and initializes it to 0
wid - the number of the warp in the block
"""
function createAndInitializeShmem(wid, threadId,sliceEdgeLength,amountOfWarps)
   #shared memory for  stroing intermidiate data per lane  
   shmem = @cuStaticSharedMem(UInt8, (513,3))
   #for storing results from warp reductions
   shmemSum = @cuStaticSharedMem(UInt16, (33,3))
    #setting shared memory to 0 
    shmem[threadId, 3]=0
    shmem[threadId, 2]=0
    shmem[threadId, 1]=0
    shmemSum[wid,1]=0
    shmemSum[wid,2]=0
    shmemSum[wid,3]=0
return (shmem,shmemSum )

end#createAndInitializeShmem


"""
reduction across the warp and adding to appropriate spots in the  shared memory
"""
function firstReduce(shmem,shmemSum,wid,threadIdx,lane   )
    @inbounds sumFn = reduce_warp(shmem[threadIdx,1],32)
    @inbounds sumFp = reduce_warp(shmem[threadIdx,2],32)
    @inbounds sumTp = reduce_warp(shmem[threadIdx,3],32)
 
    if(lane==1)
   @inbounds shmemSum[wid,1]= sumFn
     end  
    if(lane==2) 
       @inbounds shmemSum[wid,2]= sumFp
    end     
    if(lane==3)
         @inbounds shmemSum[wid,3]= sumTp
     end#if  
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
        if(numb==2 && shmemSum[33,numb]>0)
            @cuprint " blockId  $(blockId) vall $(shmemSum[33,numb]) \n"
        end
        @inbounds intermediateRes[blockId]=shmemSum[33,numb]
      end    
    #   if(lane==3)
    #     #ovewriting the value 
    #     @inbounds shmemSum[1,numb]=vall
    #   end     

  end  

end#getSecondBlockReduce


"""
generally  we want to get one block per slice as dimensions of slices in the medical images are friendly - so 256x256 ; 512x512 and 1024x1024 - it should all be possible
In order to avoid edge cases we will keep number of threads to some even multiply of 32 for ecxample 512
    arguments
        threadnum - number of threads per block
        slicesNumber - number of slices of our data
        sliceEdgeLength - slices are squares of 256, 512 or 1024 length
    output
        blockNum - number of blocks we need  - generally will return number of slices
        loopNumb - number of  iteration of each single line it need to do so a single block will cover whole slice
        indexCorr - as one lane will get access to multiple data elements we need to take correction for it 
"""
function getKernelContants(threadnum::Int,slicesNumber::Int,sliceEdgeLength::Int  )

blockNum= slicesNumber
indexCorr =  Int64(round(sliceEdgeLength*sliceEdgeLength/threadnum))
loopNumb= Int64(indexCorr-1)

return (blockNum, loopNumb, indexCorr )
end#getKernelContants
