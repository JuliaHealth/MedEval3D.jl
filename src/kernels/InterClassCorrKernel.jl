"""
calculating intercalss correlation
"""
module InterClassCorrKernel

using Main.BasicPreds, Main.CUDAGpuUtils, CUDA
using Main.MainOverlap, Main.TpfpfnKernel
export calculateInterclassCorr

"""
calculates slicewise and global interclass correlation metric
"""
function calculateInterclassCorr(flatGold
                                ,flatSegm
                                ,mainArrayDims
                                ,sumOfGold
                                ,sumOfSegm
                                ,meanOfGoldPerSlice
                                ,meanOfSegmPerSlice
                                ,sswTotal
                                ,ssbTotal
                                ,iccPerSlice
                                ,numberToLooFor)::Float64

pixelNumberPerSlice= mainArrayDims[1]*mainArrayDims[2]
loopNumb= cld(pixelNumberPerSlice,1024)

#first we need to calculate means
@cuda threads=(32,32) blocks=mainArrayDims[3]  kernel_InterClassCorr_means(flatGold,flatSegm
  ,loopNumb  ,sumOfGold,sumOfSegm ,meanOfGoldPerSlice
  ,meanOfSegmPerSlice ,pixelNumberPerSlice,numberToLooFor )

  numberOfVoxels = mainArrayDims[1]*mainArrayDims[2]*mainArrayDims[3]
  grandMean= ( (sumOfGold[1]/numberOfVoxels) + (sumOfSegm[1]/numberOfVoxels ))/2


@cuda threads=(32,32) blocks=mainArrayDims[3]  kernel_InterClassCorr(flatGold  ,flatSegm
     ,loopNumb,pixelNumberPerSlice
     ,meanOfGoldPerSlice  ,meanOfSegmPerSlice
     ,sswTotal ,ssbTotal  ,iccPerSlice     ,grandMean,numberToLooFor  )

     ssw = sswTotal[1]/numberOfVoxels;
     ssb = ssbTotal[1]/(numberOfVoxels-1) * 2;
    
  return (ssb - ssw)/(ssb + ssw);
  
end


"""
adapted from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/src/mapreduce.jl
goldBoolGPU - array holding data of gold standard bollean array
segmBoolGPU - boolean array with the data we want to compare with gold standard
ic - holding single value for global interclass correlation
intermediateRes- array holding slice wise results for ic
loopNumb - number of times the single lane needs to loop in order to get all needed data
sliceEdgeLength - length of edge of the slice we need to square this number to get number of pixels in a slice
amountOfWarps - how many warps we can stick in the block
"""

function kernel_InterClassCorr_means(flatGold
                               ,flatSegm
                                ,loopNumb::Int64
                                ,sumOfGold
                                ,sumOfSegm
                                ,meanOfGoldPerSlice
                                ,meanOfSegmPerSlice
                                ,pixelNumberPerSlice
                                ,numberToLooFor )
    #offset for lloking for values in source arrays 
    offset = (pixelNumberPerSlice*(blockIdx().x-1))
   #for storing results from warp reductions
   shmemSum = @cuStaticSharedMem(UInt16, (33,2))   #thread local values that are meant to store some results - like means ... 
   offsetIter = UInt8(1)

   locValA = UInt32(0)
   locValB = UInt32(0)
        #reset shared memory
        @ifY 1 shmemSum[threadIdxX(),1]=0 ;   @ifY 2 shmemSum[threadIdxX(),2]=0
        sync_threads()

        #first we add 1 for each spot we have true - so we will get sum  - and from sum we can get mean
        @unroll for k in 0:loopNumb
            if(threadIdxX()+(threadIdxY()-1)*32+k*1024 <=pixelNumberPerSlice)
            ind =offset+ threadIdxX()+(threadIdxY()-1)*32+k*1024
            locValA += flatGold[ind]==numberToLooFor  
            locValB += flatSegm[ind]==numberToLooFor
            end#if 
        end#for
        

            #now we will have sum of all entries in given slice that comply to our predicate
            #next we need to reduce values
            offsetIter = UInt8(1)
            while(offsetIter <32) 
                @inbounds locValA+=shfl_down_sync(FULL_MASK, locValA, offsetIter)  
                @inbounds locValB+=shfl_down_sync(FULL_MASK, locValB, offsetIter)  
                offsetIter<<= 1
            end
        # now we have sums in first threads of the warp we need to pass it to shared memory
        if(threadIdxX()==1)

            @inbounds shmemSum[threadIdxY(),1]+=locValA
            @inbounds shmemSum[threadIdxY(),2]+=locValB
        end
        sync_threads()
        #finally reduce from shared memory
        if(threadIdxY()==1)
            offsetIter = UInt8(1)
            while(offsetIter <32) 
                @inbounds shmemSum[threadIdxX(),1]+=shfl_down_sync(FULL_MASK, shmemSum[threadIdxX(),1], offsetIter)  
                @inbounds shmemSum[threadIdxX(),2]+=shfl_down_sync(FULL_MASK, shmemSum[threadIdxX(),2], offsetIter)  
            offsetIter<<= 1
            end
        end  
sync_threads()
       
      #now in   shmemSum[1,1] we should have sum of values complying with our predicate in gold mask and in shmemSum[1,2] values of other mask
      #we need to add now those to the globals  
      @ifXY 1 1  @inbounds @atomic sumOfGold[]+= shmemSum[1,1]
      @ifXY 1 2  @inbounds @atomic sumOfSegm[]+= shmemSum[1,2]
      @ifXY 1 3  @inbounds meanOfGoldPerSlice[blockIdxX()]=(shmemSum[1,1]/pixelNumberPerSlice )
      @ifXY 1 4  @inbounds meanOfSegmPerSlice[blockIdxX()]=(shmemSum[1,2]/pixelNumberPerSlice)

    return nothing
end


function kernel_InterClassCorr(flatGold
    ,flatSegm
     ,loopNumb::Int64
     ,pixelNumberPerSlice::Int64
     ,meanOfGoldPerSlice
     ,meanOfSegmPerSlice
     ,sswTotal
     ,ssbTotal
     ,iccPerSlice
     ,grandMean
     ,numberToLooFor)
  
    #offset for lloking for values in source arrays 
    offset = (pixelNumberPerSlice*(blockIdx().x-1))
    #for storing results from warp reductions
    shmemSum = @cuStaticSharedMem(Float32, (33,2))   #thread local values that are meant to store some results - like means ... 
    @ifY 1 shmemSum[threadIdxX(),1]=0 ;   @ifY 2 shmemSum[threadIdxX(),2]=0 ;@ifY 3 shmemSum[threadIdxX(),2]=0
    sync_threads()
    ssw::Float32 = Float32(0.0)
    ssb::Float32 = Float32(0.0)

    @unroll for k in UInt16(0):loopNumb
        if(threadIdxX()+(threadIdxY()-1)*32+k*1024 <=pixelNumberPerSlice)
        ind =offset+ threadIdxX()+(threadIdxY()-1)*32+k*1024   
        m =  ((flatGold[ind]==numberToLooFor) +(flatSegm[ind]==numberToLooFor))/2  
        ssw += (((flatGold[ind]==numberToLooFor)- m)^2) +(((flatSegm[ind]==numberToLooFor)- m )^2) 
        ssb += ((m - grandMean[1])^2)
        end#if 
    end#for
    #now we accumulated ssw and ssb - we need to reduce it
    offsetIter = UInt8(1)
    while(offsetIter <32) 
        @inbounds ssw+=shfl_down_sync(FULL_MASK, ssw, offsetIter)  
        @inbounds ssb+=shfl_down_sync(FULL_MASK, ssb, offsetIter)  
        offsetIter<<= 1
    end
        # now we have sums in first threads of the warp we need to pass it to shared memory
        if(threadIdxX()==1)
            @inbounds shmemSum[threadIdxY(),1]+=ssw
            @inbounds shmemSum[threadIdxY(),2]+=ssb
        end
    sync_threads()
    #finally reduce from shared memory
        if(threadIdxY()==1)
            offsetIter = UInt8(1)
            while(offsetIter <32) 
            @inbounds shmemSum[threadIdxX(),1]+=shfl_down_sync(FULL_MASK, shmemSum[threadIdxX(),1], offsetIter)  
            @inbounds shmemSum[threadIdxX(),2]+=shfl_down_sync(FULL_MASK, shmemSum[threadIdxX(),2], offsetIter)  
            offsetIter<<= 1
            end
    end  
    sync_threads()
      #now in   shmemSum[1,1] we should have ssw and in  shmemSum[1,2] ssb
      @ifXY 1 1  @inbounds @atomic sswTotal[]+= shmemSum[1,1]
      @ifXY 1 2  @inbounds @atomic ssbTotal[]+= shmemSum[1,2]
      @ifXY 1 3  @inbounds iccPerSlice[blockIdxX()]=(shmemSum[1,2] - shmemSum[1,1])/(shmemSum[1,2] + shmemSum[1,1])    
    # # ####### now we have ssw and ssb calculated both global and per slice

    return nothing


end




end#InterClassCorrKernel