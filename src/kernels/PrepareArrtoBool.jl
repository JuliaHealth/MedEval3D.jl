"""
this kernel will prepare da
"""
module PrepareArrtoBool

using CUDA, Main.GPUutils, Logging,StaticArrays



"""
This will prepare data for more complex distance metrics - we need to change input data type into boolean and find smallest possible cube that hold all necessery data

returning the data  from a kernel that  calclulate number of true positives,
true negatives, false positives and negatives par image and per slice in given data 
goldBoolGPU - array holding data of gold standard bollean array
segmBoolGPU - array with the data we want to compare with gold standard

reducedGold - the smallest boolean block (3 dim array) that contains all positive entris from both masks
reducedSegm - the smallest boolean block (3 dim array) that contains all positive entris from both masks
numberToLooFor - number we will analyze whether is the same between two sets
threadNumPerBlock - how many threads should be associated with single block
"""
function getBoolCube!(goldBoolGPU
    ,segmBoolGPU
    ,pixelNumberPerSlice::Int64
    ,numberOfSlices::Int64
    ,numberToLooFor::T
    ,IndexesArray
    ,threadNumPerBlock::Int64 = 512) where T

# we prepare the boolean array of dimensions at the begining the same as the gold standard array - later we will work only on view of it

goldDims=size(goldBoolGPU) 
reducedGold= CUDA.zeros(Bool,goldDims)
reducedSegm =CUDA.zeros(Bool,goldDims)

loopNumb, indexCorr = getKernelContants(threadNumPerBlock,pixelNumberPerSlice)
args = (goldBoolGPU
        ,segmBoolGPU
        ,tp,tn,fp,fn
        ,intermediateResTp
        ,intermediateResFp
        ,intermediateResFn
        ,loopNumb
        ,indexCorr
        ,Int64(round(threadNumPerBlock/32))
        ,pixelNumberPerSlice
        ,numberToLooFor
        ,IndexesArray
)
#getMaxBlocksPerMultiproc(args, getBlockTpFpFn) -- evaluates to 3

@cuda threads=threadNumPerBlock blocks=numberOfSlices getBlockTpFpFn(args...) 
return args
end#getTpfpfnData

"""
we need to give back number of false positive and false negatives and min,max x,y,x of block containing all data 
IMPORTANT - in order to avoid bound checking on every iteration we need to keep the dimension of the resulting block be divided by data block cube size for example 32
IMPORTANT - we assume that x dim can not be bigger than 1024
adapted from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/src/mapreduce.jl
goldBoolGPU3d - array holding 3 dimensional data of gold standard bollean array
segmBoolGPU3d - array with 3 dimensional the data we want to compare with gold standard
reducedGold - the smallest boolean block (3 dim array) that contains all positive entris from both masks
reducedSegm - the smallest boolean block (3 dim array) that contains all positive entris from both masks
numberToLooFor - number we will analyze whether is the same between two sets
loopNumbYdim - number of times the single lane needs to loop in order to get all needed data - in this kernel it will be exactly a y dimension of a slice
xdim - length in x direction of source array 
loopNumbXdim - in case the x dim will be bigger than number of threads we will create second inner loop
cuda arrays holding just single value wit atomically added result
,fn,fp
,minxRes,maxxRes
,minyRes,maxyRes
,minZres,maxZres

"""
function getBoolCubeKernel(goldBoolGPU3d
        ,segmBoolGPU3d
        ,reducedGold
        ,reducedSegm
        ,loopNumbYdim::UInt16
        ,indexCorr::UInt16
        ,pixelNumberPerSlice::UInt16
        ,xdim::UInt16
        ,loopNumbXdim::UInt16
        ,numberToLooFor::T
        ,IndexesArray
        ,fn,fp
        ,minxRes,maxxRes
        ,minyRes,maxyRes
        ,minZres,maxZres
) where T
    # we multiply thread id as we are covering now 2 places using one lane - hence after all lanes gone through we will cover 2 blocks - hence second multiply    
   #i = correctedIdx + ((blockIdx().x - 1) *indexCorr) * (blockDim().x)# used as a basis to get data we want from global memory
   wid, lane = getWidAndLane(threadIdx().x)

#creates shared memory and initializes it to 0
   shmemSum = createAndInitializeShmem(wid,threadIdx().x,amountOfWarps,lane)
# incrementing appropriate number of times 
   
    #0 - false negative; 1- false positive; 2 -minx; 3 max x; 4 miny; 5 maxy
    locArr= zeros(MVector{6,UInt16})
    
    @unroll for k in 0:loopNumbYdim# k is effectively y dimension
        for kx in 0:loopNumbXdim
            if(threadIdx().x<=xdim)           
                incr_locArr(goldBoolGPU[threadIdx().x+kx*xdim, k+1, blockIdx().x]==numberToLooFor
                            ,segmBoolGPU[threadIdx().x+kx*xdim, k+1, blockIdx().x]==numberToLooFor
                            ,locArr
                            ,threadIdx().x+kx*xdim
                            ,k+1
                            ,blockIdx().x
                            ,reducedGold
                            ,reducedSegm)
            end#if 
        end#for 
        
    end#for

    firstReduce(locArr,shmemSum)

    

    sync_threads()
    #now all data about of intrest should be in  shared memory so we will get all rsults from warp reduction in the shared memory 
    #locArr  0 - false negative; 1- false positive; 2 -minx; 3 max x; 4 miny; 5 maxy for shmem sum we need to add 1 as shmem is 1 based

    getSecondBlockReduceSum( 1,1,wid,fn,shmemSum,blockIdx().x,lane)
    getSecondBlockReduceSum( 2,2,wid,fp,shmemSum,blockIdx().x,lane)

    getSecondBlockReduceMin( 3,3,wid,minxRes,shmemSum,blockIdx().x,lane)
    getSecondBlockReduceMax( 4,4,wid,maxxRes,shmemSum,blockIdx().x,lane)
    getSecondBlockReduceMin( 5,5,wid,minyRes,shmemSum,blockIdx().x,lane)
    getSecondBlockReduceMax( 6,6,wid,maxxRes,shmemSum,blockIdx().x,lane)

   return  
   end

"""
add value to the shared memory in the position i, x where x is 1 ,2 or 3 and is calculated as described below
boolGold & boolSegm + boolGold +1 will evaluate to 
    ⊻- xor gate 
    1 in case of false negative
    2 in case of false positive
x,y,z - the coordinates we are currently in 

"""
@inline function incr_locArr(boolGold::Bool
                            ,boolSegm::Bool
                            ,locArr::MVector{6, UInt16}
                            ,x,y,z
                            ,reducedGold
                            ,reducedSegm)
    #first we need the flase positives and false negatives - this will write also true positive - but later we will 
    @inbounds locArr[boolGold+ boolSegm+ boolSegm]+=(boolGold  ⊻ boolSegm)
    #in case some is positive we can go futher with looking for max,min in dims and add to the new reduced boolean arrays waht we are intrested in  
    if(boolGold  || boolSegm)
    #locArr  0 - false negative; 1- false positive; 2 -minx; 3 max x; 4 miny; 5 maxy
    locArr[2]= min(locArr[3],x)
    locArr[3]= max(locArr[4],x)
    locArr[4]= min(locArr[5],y)
    locArr[5]= max(locArr[6],y)
    reducedGold[x,y,z]=true    
    reducedSegm[x,y,z]=true    

    end    
   
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
function createAndInitializeShmem(wid, threadId,amountOfWarps,lane)
   #for storing results from warp reductions
   shmemSum = @cuStaticSharedMem(UInt16, (33,6))

    if(wid==1)
        shmemSum[lane,1]=0
    elseif(wid==2)
        shmemSum[lane,2]=0
    elseif(wid==3)
        shmemSum[lane,3]=0     
    elseif(wid==4)
        shmemSum[lane,5]=0       
    elseif(wid==5)
        shmemSum[lane,5]=0            
    elseif(wid==6)
        shmemSum[lane,6]=0
    end            

return shmemSum

end#createAndInitializeShmem


"""
reduction across the warp and adding to appropriate spots in the  shared memory
"""
function firstReduce(locArr,shmemSum)
    #locArr  0 - false negative; 1- false positive; 2 -minx; 3 max x; 4 miny; 5 maxy
    @inbounds shmemSum[wid,1] = reduce_warp(locArr[0],32)
    @inbounds shmemSum[wid,2] = reduce_warp(locArr[1],32)

    @inbounds shmemSum[wid,3] = reduce_warp_min(locArr[2],32)
    @inbounds shmemSum[wid,4] = reduce_warp_max(locArr[3],32)
    @inbounds shmemSum[wid,5] = reduce_warp_min(locArr[4],32)
    @inbounds shmemSum[wid,6] = reduce_warp_max(locArr[5],32)



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
function getSecondBlockReduceSum(chosenWid,numb,wid, singleREs,shmemSum,blockId,lane)
    if(wid==chosenWid )
        shmemSum[33,numb] = reduce_warp(shmemSum[lane,numb],32 )
        
      #probably we do not need to sync warp as shfl dow do it for us         
      if(lane==1)
          @inbounds @atomic singleREs[]+=shmemSum[33,numb]    
      end    
    #   if(lane==3)
    #     #ovewriting the value 
    #     @inbounds shmemSum[1,numb]=vall
    #   end     

  end  


end#getSecondBlockReduce
function getSecondBlockReduceMin(chosenWid,numb,wid, singleREs,shmemSum,blockId,lane)
    if(wid==chosenWid )
        shmemSum[33,numb] = reduce_warp_min(shmemSum[lane,numb],32 )
        

      #probably we do not need to sync warp as shfl dow do it for us         
      if(lane==1)
          @inbounds @atomic singleREs[]=min(shmemSum[33,numb],singleREs[])    
      end    
    #   if(lane==3)
    #     #ovewriting the value 
    #     @inbounds shmemSum[1,numb]=vall
    #   end     

  end  

end#getSecondBlockReduce
function getSecondBlockReduceMax(chosenWid,numb,wid, singleREs,shmemSum,blockId,lane)
    if(wid==chosenWid )
        shmemSum[33,numb] = reduce_warp_max(shmemSum[lane,numb],32 )
        
      #probably we do not need to sync warp as shfl dow do it for us         
      if(lane==1)
          @inbounds @atomic singleREs[]=max(shmemSum[33,numb],singleREs[])     
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
# correctedIdx = (threadIdx().x-1)* indexCorr+1
# i= correctedIdx
# #i = correctedIdx + ((blockIdx().x - 1) *indexCorr) * (blockDim().x)# used as a basis to get data we want from global memory
# wid, lane = fldmod1(threadIdx().x,32)
# #creates shared memory and initializes it to 0
# shmem,shmemSum = createAndInitializeShmem()
# shmem[513,1]= numberToLooFor
# ##### in this outer loop we are iterating over all slices that this block is responsible for
# @unroll for blockRef in 1:maxSlicesPerBlock    
#     sliceNumb= slicesPerBlockMatrix[blockIdx().x,blockRef]
#         if(sliceNumb>0)
#             i = correctedIdx + (pixelNumberPerSlice*(sliceNumb-1))# used as a basis to get data we want from global memory
#             setShmemTo0(wid,threadIdx().x,lane,shmem,shmemSum)           
#             # incrementing appropriate number of times 
        
#         @unroll for k in 0:loopNumb
#                 if(correctedIdx+k<=pixelNumberPerSlice)
#                     incr_shmem(threadIdx().x,goldBoolGPU[i+k]==shmem[513,1],segmBoolGPU[i+k]==shmem[513,1],shmem)
#                 end#if
#             end#for   
#         #reducing across the warp
#         firstReduce(shmem,shmemSum,wid,threadIdx().x,lane,IndexesArray,i)
        
        
#         sync_threads()
#         #now all data about of intrest should be in  shared memory so we will get all rsults from warp reduction in the shared memory 
#         getSecondBlockReduce( 1,3,wid,intermediateResTp,tp,shmemSum,blockIdx().x,lane)
#         getSecondBlockReduce( 2,2,wid,intermediateResFp,fp,shmemSum,blockIdx().x,lane)
#         getSecondBlockReduce( 3,1,wid,intermediateResFn,fn,shmemSum,blockIdx().x,lane)
#     end#if     
# end#for

# return  
# end
