"""
Holding developed kernels  in progression when new optimazation is povided
in order to be able to benchmark each on given data 
"""
module BasicPreds
using CUDA
export getBigTestBools,getSmallTestBools,getExampleKernelArgs


function getSmallTestBools()

    nx=512
    ny=512
    nz=317

    #first we initialize the metrics on CPU so we will modify them easier
    goldBool= zeros(Float32,nx,ny,nz); #mimicks gold standard mask
    segmBool= zeros(Float32,nx,ny,nz); #mimicks mask     
# so we  have 2 cubes that are overlapped in their two thirds
    cartTrueGold =  CartesianIndices(zeros(3,3,5) ).+CartesianIndex(5,5,5);
    cartTrueSegm =  CartesianIndices(zeros(3,3,3) ).+CartesianIndex(4,5,5); 
    goldBool[cartTrueGold].=1.0
    segmBool[cartTrueSegm].=1.0



    cartTrue =  CartesianIndices(zeros(9,9,9) ).+CartesianIndex(80,80,80);
    cartTrueB =  CartesianIndices(zeros(9,9,9) ).+CartesianIndex(200,200,200);

    goldBool[cartTrue].=1.0
    segmBool[cartTrue].=1.0


    #for storing output total first tp than TN than Fp and Fn
    tp= CuArray([0]);
    tn= CuArray([0]);
    fp= CuArray([0]);
    fn = CuArray([0]);
    
    #for storing metrics for slice
    tpArr = CuArray(zeros(Int16,nz));
    tnArr = CuArray(zeros(Int16,nz));
    fpArr = CuArray(zeros(Int16,nz));
    fnArr = CuArray(zeros(Int16,nz));
### calculating correct results (unoptimazied way) just for unit testing

# FlattG = vec(goldBool);
# FlattSeg = vec(segmBool);
ff = zeros(Float32,1000)
FlattG = vcat(vec(goldBool),ff)
FlattSeg = vcat(vec(segmBool),ff)


FlattGoldGPU= CuArray( FlattG)
FlattSegGPU= CuArray( FlattSeg )
# total for all slices
# tpTotalTrue = filter(pair->pair[2]== FlattB[pair[1]] ==true ,collect(enumerate(FlattG)))|>length
# tnTotalTrue = filter(pair->pair[2]== FlattB[pair[1]] ==false ,collect(enumerate(FlattG)))|>length
# fpTotalTrue = filter(pair->!pair[2] && FlattB[pair[1]] ,collect(enumerate(FlattG)))|>length
# fnTotalTrue = filter(pair->pair[2] && !FlattB[pair[1]] ==true ,collect(enumerate(FlattG)))|>length

# correct result per slice 

# toIterSlices =  collect(enumerate(collect(eachslice(goldBool, dims = 3)))) 
# tpPerSliceTrue =   map(slicePair->   filter( pair->  pair[2]== vec(segmBool[:,:,slicePair[1]])[pair[1]] ==true         ,  collect(enumerate(vec(collect(slicePair[2])))) )|>length  , toIterSlices   )
# tnPerSliceTrue =  map(slicePair->   filter( pair->  pair[2]== vec(segmBool[:,:,slicePair[1]])[pair[1]] ==false         ,  collect(enumerate(vec(collect(slicePair[2])))) )|>length  , toIterSlices   )
# fpPerSliceTrue =    map(slicePair->   filter( pair->  !pair[2] && vec(segmBool[:,:,slicePair[1]])[pair[1]]       ,  collect(enumerate(vec(collect(slicePair[2])))) )|>length  , toIterSlices   )
# fnPerSliceTrue =   map(slicePair->   filter( pair->  pair[2] && !vec(segmBool[:,:,slicePair[1]])[pair[1]]          ,  collect(enumerate(vec(collect(slicePair[2])))) )|>length  , toIterSlices   )
# #sum for all dummy image
# tpTotalTrue=  tpPerSliceTrue|>sum 
# fpTotalTrue= fpPerSliceTrue|>sum 
# fnTotalTrue= fnPerSliceTrue|>sum 


toIterSlices = []#collect(enumerate(collect(eachslice(goldBool, dims = 3)))) 
tpPerSliceTrue = []# map(slicePair->   filter( pair->  pair[2]== vec(segmBool[:,:,slicePair[1]])[pair[1]] ==true         ,  collect(enumerate(vec(collect(slicePair[2])))) )|>length  , toIterSlices   )
tnPerSliceTrue = []#map(slicePair->   filter( pair->  pair[2]== vec(segmBool[:,:,slicePair[1]])[pair[1]] ==false         ,  collect(enumerate(vec(collect(slicePair[2])))) )|>length  , toIterSlices   )
fpPerSliceTrue = []#  map(slicePair->   filter( pair->  !pair[2] && vec(segmBool[:,:,slicePair[1]])[pair[1]]       ,  collect(enumerate(vec(collect(slicePair[2])))) )|>length  , toIterSlices   )
fnPerSliceTrue = []# map(slicePair->   filter( pair->  pair[2] && !vec(segmBool[:,:,slicePair[1]])[pair[1]]          ,  collect(enumerate(vec(collect(slicePair[2])))) )|>length  , toIterSlices   )
#sum for all dummy image
tpTotalTrue= 747# tpPerSliceTrue|>sum 
fpTotalTrue= 9#fpPerSliceTrue|>sum 
fnTotalTrue= 27#fnPerSliceTrue|>sum 


tnTotalTrue= (nx*ny*nz)-tpTotalTrue-fpTotalTrue-fnTotalTrue

goldBoolGPU= CuArray( goldBool)
segmBoolGPU= CuArray( segmBool )




blockNum = Int64(round(length(FlattG)/1024))

# array needs to hold 3 values tp, fp and false negatives from each block

intermediateResTp = CUDA.zeros(Int32, blockNum+2)
intermediateResFp = CUDA.zeros(Int32, blockNum+2)
intermediateResFn = CUDA.zeros(Int32, blockNum+2)

#intermediateResults = CUDA.zeros(Int32, Int64(((nx*ny*nz)/32)+100)  , 3)

    ## so there should be 9  true positives, 

# returning bunch of values so writing all will be simpler
    return (goldBoolGPU,segmBoolGPU,tp,tn,fp,fn, tpArr,tnArr,fpArr, fnArr, blockNum , nx,ny,nz,tpTotalTrue,tnTotalTrue,fpTotalTrue, fnTotalTrue ,tpPerSliceTrue,  tnPerSliceTrue,fpPerSliceTrue,fnPerSliceTrue ,FlattG, FlattSeg ,FlattGoldGPU,FlattSegGPU,intermediateResTp,intermediateResFp,intermediateResFn)
    
    end
    





"""
most primitive working example
"""
function primitiveAtomicKernel(goldBoolGPU::CuDeviceArray{Bool,1, 1}, segmBoolGPU::CuDeviceArray{Bool, 1, 1},tp,tn,fp, fn,nx,ny,nz,xthreads, ythreads,zthreads)
        # getting all required indexes
        i = threadIdxX() + (blockIdx().x - 1) * blockDimX()
        #if(i< (nx*ny*nz)/ythreads ) #i<nx*ny && j<ny && z<nz
       # CUDA.@cuprint "goldBoolGPU[i,j,z] $(goldBoolGPU[i,j,z]) segmBoolGPU[i,j,z] $(segmBoolGPU[i,j,z]) i $(i) j $(j) z $(z) "
            if(goldBoolGPU[i] & segmBoolGPU[i] )
                @atomic tp[]+=1
            elseif (!goldBoolGPU[i] & !segmBoolGPU[i] )
                @atomic tn[]+=1
            elseif (!goldBoolGPU[i] & segmBoolGPU[i] )
                @atomic fp[]+=1    
            elseif (goldBoolGPU[i] & !segmBoolGPU[i] )
                @atomic fn[]+=1    
            end
        #else
         #   CUDA.@cuprint "i $(i) j $(j) z $(z)  \n"
    
    
        return  
    
        end



        

 end   


 



 """

 
 julia> BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true
 true
 
 julia> function toBench(goldBoolGPU,segmBoolGPU,tp,tn,fp,fn)
            CUDA.@sync TpfpfnKernel.getTpfpfnData!(arrGold,arrAlgo,tp,tn,fp,fn , intermediateResTp,intermediateResFp ,intermediateResFn,sizz[1]*sizz[2],sizz[3],UInt8(1),IndexesArray)
        end
 toBench (generic function with 1 method)
 julia> bb = @benchmark toBench(goldBoolGPU,segmBoolGPU,tp,tn,fp,fn)  setup=(goldBoolGPU,segmBoolGPU,tp,tn,fp,fn, tpArr,tnArr,fpArr, fnArr, blockNum , nx,ny,nz ,tpTotalTrue,tnTotalTrue,fpTotalTrue, fnTotalTrue ,tpPerSliceTrue,  tnPerSliceTrue,fpPerSliceTrue,fnPerSliceTrue ,flattG, flattSeg ,FlattGoldGPU,FlattSegGPU,intermediateResTp,intermediateResFp,intermediateResFn = getSmallTestBools())
 BenchmarkTools.Trial: 100 samples with 1 evaluation.
  Range (min … max):  823.000 μs …   1.546 ms  ┊ GC (min … max): 0.00% … 0.00%
  Time  (median):     831.800 μs               ┊ GC (median):    0.00%
  Time  (mean ± σ):   860.136 μs ± 123.951 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%
 """

#  function getTpfpfnData!(goldBoolGPU
#     , segmBoolGPU
#     ,tp,tn,fp,fn
#     ,intermediateResTp
#     ,intermediateResFp
#     ,intermediateResFn
#     ,pixelNumberPerSlice::Int64
#     ,numberOfSlices::Int64
#     ,numberToLooFor::T
#     ,IndexesArray

#     ,threadNumPerBlock::Int64 = 512) where T


 

# loopNumb, indexCorr = getKernelContants(threadNumPerBlock,pixelNumberPerSlice)
# args = (goldBoolGPU
#         ,segmBoolGPU
#         ,tp,tn,fp,fn
#         ,intermediateResTp
#         ,intermediateResFp
#         ,intermediateResFn
#         ,loopNumb
#         ,indexCorr
#         ,Int64(round(threadNumPerBlock/32))
#         ,pixelNumberPerSlice
#         ,numberToLooFor
#         ,IndexesArray
# )
# #getMaxBlocksPerMultiproc(args, getBlockTpFpFn) -- evaluates to 3

# @cuda threads=threadNumPerBlock blocks=numberOfSlices getBlockTpFpFn(args...) 
# return args
# end#getTpfpfnData

# """
# adapted from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/src/mapreduce.jl
# goldBoolGPU - array holding data of gold standard bollean array
# segmBoolGPU - boolean array with the data we want to compare with gold standard
# tp,tn,fp,fn - holding single values for true positive, true negative, false positive and false negative
# intermediateResTp, intermediateResFp, intermediateResFn - arrays holding slice wie results for true positive ...
# loopNumb - number of times the single lane needs to loop in order to get all needed data
# sliceEdgeLength - length of edge of the slice we need to square this number to get number of pixels in a slice
# amountOfWarps - how many warps we can stick in the vlock
# """
# function getBlockTpFpFn(goldBoolGPU
#         , segmBoolGPU
#         ,tp,tn,fp,fn
#         ,intermediateResTp
#         ,intermediateResFp
#         ,intermediateResFn
#         ,loopNumb::Int64
#         ,indexCorr::Int64
#         ,amountOfWarps::Int64
#         ,pixelNumberPerSlice::Int64
#         ,numberToLooFor::T
#         ,IndexesArray
# ) where T
#     # we multiply thread id as we are covering now 2 places using one lane - hence after all lanes gone through we will cover 2 blocks - hence second multiply    
#     correctedIdx = (threadIdxX()-1)* indexCorr+1
#     i = correctedIdx + (pixelNumberPerSlice*(blockIdx().x-1))
#     #i = correctedIdx + ((blockIdx().x - 1) *indexCorr) * (blockDimX())# used as a basis to get data we want from global memory
#    wid, lane = fldmod1(threadIdxX(),32)
# #creates shared memory and initializes it to 0
#    shmem,shmemSum = createAndInitializeShmem(wid,threadIdxX(),amountOfWarps,lane)
#    shmem[513,1]= numberToLooFor
# # incrementing appropriate number of times 

#     @unroll for k in 0:loopNumb
#     if(correctedIdx+k<=pixelNumberPerSlice)
#         incr_shmem(threadIdxX(),goldBoolGPU[i+k]==shmem[513,1],segmBoolGPU[i+k]==shmem[513,1],shmem)
#         #incr_shmem(threadIdxX()+1,goldBoolGPU[i+k]==shmem[513,1],segmBoolGPU[i+k]==shmem[513,1],shmem,IndexesArray)
#     end    
#     end#for

#     #reducing across the warp
#     firstReduce(shmem,shmemSum,wid,threadIdxX(),lane,IndexesArray,i)

#     sync_threads()
#     #now all data about of intrest should be in  shared memory so we will get all rsults from warp reduction in the shared memory 
#     getSecondBlockReduce( 1,3,wid,intermediateResTp,tp,shmemSum,blockIdx().x,lane)
#     getSecondBlockReduce( 2,2,wid,intermediateResFp,fp,shmemSum,blockIdx().x,lane)
#     getSecondBlockReduce( 3,1,wid,intermediateResFn,fn,shmemSum,blockIdx().x,lane)

#    return  
#    end

# """
# add value to the shared memory in the position i, x where x is 1 ,2 or 3 and is calculated as described below
# boolGold & boolSegm + boolGold +1 will evaluate to 
#     3 in case  of true positive
#     2 in case of false positive
#     1 in case of false negative
# """
# @inline function incr_shmem( primi::Int64,boolGold::Bool,boolSegm::Bool,shmem )
#     @inbounds shmem[ primi, (boolGold & boolSegm + boolSegm +1) ]+=(boolGold | boolSegm) 
#     return true
# end


# """
# creates shared memory and initializes it to 0
# wid - the number of the warp in the block
# """
# function createAndInitializeShmem(wid, threadId,amountOfWarps,lane)
#    #shared memory for  stroing intermidiate data per lane  
#    shmem = @cuStaticSharedMem(UInt16, (513,3))
#    #for storing results from warp reductions
#    shmemSum = @cuStaticSharedMem(UInt16, (33,3))
#     #setting shared memory to 0 
#     shmem[threadId, 3]=0
#     shmem[threadId, 2]=0
#     shmem[threadId, 1]=0
    
#     if(wid==1)
#     shmemSum[lane,1]=0
#     end
#     if(wid==2)
#         shmemSum[lane,2]=0
#     end
#     if(wid==3)
#     shmemSum[lane,3]=0
#     end            

# return (shmem,shmemSum )

# end#createAndInitializeShmem


# """
# reduction across the warp and adding to appropriate spots in the  shared memory
# """
# function firstReduce(shmem,shmemSum,wid,threadIdx,lane,IndexesArray,i   )
#     @inbounds shmemSum[wid,1] = reduce_warp(shmem[threadIdx,1],32)
#     @inbounds shmemSum[wid,2] = reduce_warp(shmem[threadIdx,2],32)
#     @inbounds shmemSum[wid,3] = reduce_warp(shmem[threadIdx,3],32)
# end#firstReduce

# """
# sets the final block amount of true positives, false positives and false negatives and saves it
# to the  array representing each slice, 
# wid - the warp in a block we want to use
# numb - number associated with constant - used to access shared memory for example
# chosenWid - on which block we want to make a reduction to happen
# intermediateRes - array with intermediate -  slice wise results
# singleREs - the final  constant holding image witde values (usefull for example for debugging)
# shmemSum - shared memory where we get the  results to be reduced now and to which we will also save the output
# blockId - number related to block we are currently in 
# lane - the lane in the warp
# """
# function getSecondBlockReduce(chosenWid,numb,wid, intermediateRes,singleREs,shmemSum,blockId,lane)
#     if(wid==chosenWid )
#         shmemSum[33,numb] = reduce_warp(shmemSum[lane,numb],32 )
        
#       #probably we do not need to sync warp as shfl dow do it for us         
#       if(lane==1)
#           @inbounds @atomic singleREs[]+=shmemSum[33,numb]
#       end    
#       if(lane==2)

#         @inbounds intermediateRes[blockId]=shmemSum[33,numb]
#       end    
#     #   if(lane==3)
#     #     #ovewriting the value 
#     #     @inbounds shmemSum[1,numb]=vall
#     #   end     

#   end  





# using CUDA

# A=rand(Bool,2,2,2)
# CuArray(view(A,: ))


# function getBigTestBools()
# goldBool= falses(128,128,10);#mimicks gold standard mask
# segmBool= falses(128,128,10);# mimicks mask 
# goldBool= falses(128,128,10);#mimicks gold standard mask
# segmBool= falses(128,128,10);# mimicks mask 
# cartTrueGold = [CartesianIndex(100,100,5), CartesianIndex(100,101,6), CartesianIndex(100,102,7)];
# cartTrueSegm = [CartesianIndex(99,100,5), CartesianIndex(100,101,6), CartesianIndex(100,102,7) ];
# CartesianIndex(100,100,5); # false negative;
# CartesianIndex(99,100,5); # false positive
# (CartesianIndex(100,101,6), CartesianIndex(100,102,7)); # true positive 
# goldBool[cartTrueGold].=true;
# segmBool[cartTrueSegm].=true;
# goldBoolGPU= CuArray(goldBool);
# segmBoolGPU= CuArray(segmBool);



# tp = CuArray([0])
# tn = CuArray([0])
# fp = CuArray([0])
# fn = CuArray([0])

# n = 128*128*10

# blockNum = Int64(ceil((n)/(8*8*8)))



# return (goldBoolGPU,segmBoolGPU,tp,tn,fp, fn,blockNum, n   )
# end





