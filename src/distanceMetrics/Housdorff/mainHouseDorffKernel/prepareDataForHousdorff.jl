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
cuda arrays holding just single value wit atomically reduced result
,fn,fp
,minxRes,maxxRes
,minyRes,maxyRes
,minZres,maxZres
"""
function getBoolCubeKernel(goldBoolGPU3d
        ,segmBoolGPU3d
        ,reducedGoldA
        ,reducedSegmA
        ,reducedGoldB
        ,reducedSegmB
        ,loopNumbYdim::UInt16
        ,xdim::UInt16
        ,loopNumbXdim::UInt16
        ,numberToLooFor::T
        ,IndexesArray
        ,fn::CuDeviceVector{UInt32, 1}
        ,fp::CuDeviceVector{UInt32, 1}
        ,minxRes::CuDeviceVector{UInt32, 1}
        ,maxxRes::CuDeviceVector{UInt32, 1}
        ,minyRes::CuDeviceVector{UInt32, 1}
        ,maxyRes::CuDeviceVector{UInt32, 1}
        ,minZres::CuDeviceVector{UInt32, 1}
        ,maxZres::CuDeviceVector{UInt32, 1}
        ,warpNumber
) where T
   
   anyPositive = false # true If any bit will bge positive in this array - we are not afraid of data race as we can set it multiple time to true
#creates shared memory and initializes it to 0
   shmemSum = createAndInitializeShmem(wid,threadIdxX(),lane)
# incrementing appropriate number of times 
   
  #0 - false negative; 1- false positive; 2 -minx; 3 max x; 4 miny; 5 maxy
  locArr= zeros(MVector{6,UInt16})
  
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


   return  
   end
