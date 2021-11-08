"""

new optimazation idea  - try to put all data in boolean arrays in shared memory  when getting means
next we would need only to read shared memory - yet first one need to check wheather there would be enough shmem on device



calculating intercalss correlation
"""
module InterClassCorrKernel

using Main.BasicPreds, Main.CUDAGpuUtils, CUDA
using Main.MainOverlap, Main.TpfpfnKernel
export calculateInterclassCorr



"""
preparation for the interclass correlation kernel - we prepare the  GPU arrays - in this kernel they are small , calculate required loop iterations 
and using occupancy API calculate optimal number of  threads and blocks for both kernels 
also return arguments lists for both kernels
"""
function prepareInterClassCorrKernel(goldlGPU,segmGPU)
  
  argsMeans= (goldlGPU,segmGPU,loopXdim,loopYdim,loopZdim ,sumOfGold,sumOfSegm ,numberToLooFor )
  
  argsMain= (goldlGPU,segmGPU  ,loopXdim,loopYdim,loopZdim ,sswTotal ,ssbTotal  ,grandMean  ,numberToLooFo)
end



"""
calculates slicewise and global interclass correlation metric

threadsForMeans, threadsForMain - tuples indicating dimensionality  of thread block 
"""
function calculateInterclassCorr(argsMeans,argsMain, threadsForMeans, threadsForMain, blocksForMeans, blocksForMain)::Float64

pixelNumberPerBlock = cld(mainArrayDims[1]*mainArrayDims[2]*mainArrayDims[3],maxBlocksPerKernel)-1 # some single pixels at the ends may be ignored
pixelNumberPerSlice = mainArrayDims[1]*mainArrayDims[2]
#first we need to calculate means
@cuda threads=threadsForMeans blocks=blocksForMeans  kernel_InterClassCorr_means(argsMeans )
  
  numberOfVoxels = mainArrayDims[1]*mainArrayDims[2]*mainArrayDims[3]
  grandMean= ( (sumOfGold[1]/numberOfVoxels) + (sumOfSegm[1]/numberOfVoxels ))/2
  # basically only grand mean needs to be updated 
  argsMain[8]=grandMean

@cuda threads=threadsForMain blocks=blocksForMain  kernel_InterClassCorr(argsMain)

     ssw = sswTotal[1]/numberOfVoxels;
     ssb = ssbTotal[1]/(numberOfVoxels-1) * 2;
    
 return (ssb - ssw)/(ssb + ssw);
 
end


"""
adapted from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/src/mapreduce.jl
goldGPU - array holding data of gold standard  array
segmGPU -  array with the data we want to compare with gold standard
ic - holding single value for global interclass correlation
intermediateRes- array holding slice wise results for ic
loopXdim,loopYdim,loopZdim - number of times we need to loop over each dimension in order to get all needed data
sliceEdgeLength - length of edge of the slice we need to square this number to get number of pixels in a slice
amountOfWarps - how many warps we can stick in the block
"""

function kernel_InterClassCorr_means(goldlGPU
                               ,segmGPU
                                ,loopXdim,loopYdim,loopZdim
                                ,sumOfGold
                                ,sumOfSegm
                                ,numberToLooFor )
  
  locValA = UInt32(0)
 locValB = UInt32(0)
  offsetIter= UInt8(1)
  shmemSum = @cuStaticSharedMem(UInt16, (33,2))   #thread local values that are meant to store some results - like means ... 
      @iter3dAdditionalzActs(arrDims,loopXdim,loopYdim,loopZdim,
    #inner expression
    begin
        #updating variables needed to calculate means
           locValA += goldlGPU[x,y,z]==numberToLooFor  
           locValB += segmGPU[z,y,z]==numberToLooFor
    end,
    #after z expression - we get slice wise true counts from it 
    begin         end ) 

    #tell what variables are to be reduced and by what operation
    @redWitAct(offsetIter,shmemSum,  locValA,+,     locValB,+  )
    @sendAtomicHelperAndAdd(shmemSum, sumOfGold, sumOfSegm)

#     #offset for lloking for values in source arrays 
#     offset = (pixelNumberPerSlice*(blockIdx().x-1))
#    #for storing results from warp reductions
#    shmemSum = @cuStaticSharedMem(UInt16, (33,2))   #thread local values that are meant to store some results - like means ... 
#    offsetIter = UInt8(1)

#    locValA = UInt32(0)
#    locValB = UInt32(0)
#         #reset shared memory
#         @ifY 1 shmemSum[threadIdxX(),1]=0 ;   @ifY 2 shmemSum[threadIdxX(),2]=0
#         sync_threads()

#         #first we add 1 for each spot we have true - so we will get sum  - and from sum we can get mean
#         @unroll for k in 0:loopNumb
#             if(threadIdxX()+(threadIdxY()-1)*32+k*1024 <=pixelNumberPerSlice)
#             ind =offset+ threadIdxX()+(threadIdxY()-1)*32+k*1024
#             locValA += flatGold[ind]==numberToLooFor  
#             locValB += flatSegm[ind]==numberToLooFor
#             end#if 
#         end#for
        

#             #now we will have sum of all entries in given slice that comply to our predicate
#             #next we need to reduce values
#             offsetIter = UInt8(1)
#             while(offsetIter <32) 
#                 @inbounds locValA+=shfl_down_sync(FULL_MASK, locValA, offsetIter)  
#                 @inbounds locValB+=shfl_down_sync(FULL_MASK, locValB, offsetIter)  
#                 offsetIter<<= 1
#             end
#         # now we have sums in first threads of the warp we need to pass it to shared memory
#         if(threadIdxX()==1)

#             @inbounds shmemSum[threadIdxY(),1]+=locValA
#             @inbounds shmemSum[threadIdxY(),2]+=locValB
#         end
#         sync_threads()
#         #finally reduce from shared memory
#         if(threadIdxY()==1)
#             offsetIter = UInt8(1)
#             while(offsetIter <32) 
#                 @inbounds shmemSum[threadIdxX(),1]+=shfl_down_sync(FULL_MASK, shmemSum[threadIdxX(),1], offsetIter)  
#                 @inbounds shmemSum[threadIdxX(),2]+=shfl_down_sync(FULL_MASK, shmemSum[threadIdxX(),2], offsetIter)  
#             offsetIter<<= 1
#             end
#         end  
# sync_threads()
       
#       #now in   shmemSum[1,1] we should have sum of values complying with our predicate in gold mask and in shmemSum[1,2] values of other mask
#       #we need to add now those to the globals  
#       @ifXY 1 1  @inbounds @atomic sumOfGold[]+= shmemSum[1,1]
#       @ifXY 1 2  @inbounds @atomic sumOfSegm[]+= shmemSum[1,2]

    return nothing
end


function kernel_InterClassCorr(goldlGPU,segmGPU 
     ,loopXdim,loopYdim,loopZdim
     ,sswTotal
     ,ssbTotal
     ,grandMean
     ,numberToLooFor )
  
  ssw = Float32(0)
  ssb = Float32(0)
  m = Float32(0)
   offsetIter= UInt8(1)
  shmemSum = @cuStaticSharedMem(Float32, (33,2))   #thread local values that are meant to store some results - like means ... 
      @iter3dAdditionalzActs(arrDims,loopXdim,loopYdim,loopZdim,
    #inner expression
    begin
        m =  ((goldlGPU[x,y,z]==numberToLooFor ) +(segmGPU[z,y,z]==numberToLooFor)/2  
        ssw += (((goldlGPU[x,y,z]==numberToLooFor)- m)^2) +(((segmGPU[z,y,z]==numberToLooFor)- m )^2) 
        ssb += ((m - grandMean[1])^2)
    end,
    #after z expression - we get slice wise true counts from it 
    begin         end ) 

    #tell what variables are to be reduced and by what operation
    @redWitAct(offsetIter,shmemSum,  ssw,+,     ssb,+  )
  sync_threads()
    @sendAtomicHelperAndAdd(shmemSum, sswTotal,ssbTotal)
  
  
#     #offset for lloking for values in source arrays 
#     offset = (pixelNumberPerSlice*(blockIdx().x-1))
#     #for storing results from warp reductions
#     shmemSum = @cuStaticSharedMem(Float32, (33,2))   #thread local values that are meant to store some results - like means ... 
#     @ifY 1 shmemSum[threadIdxX(),1]=0 ;   @ifY 2 shmemSum[threadIdxX(),2]=0 ;@ifY 3 shmemSum[threadIdxX(),2]=0
#     sync_threads()
#     ssw::Float32 = Float32(0.0)
#     ssb::Float32 = Float32(0.0)

#     @unroll for k in UInt16(0):loopNumb
#         if(threadIdxX()+(threadIdxY()-1)*32+k*1024 <=pixelNumberPerSlice)
#         ind =offset+ threadIdxX()+(threadIdxY()-1)*32+k*1024   
#         m =  ((flatGold[ind]==numberToLooFor) +(flatSegm[ind]==numberToLooFor))/2  
#         ssw += (((flatGold[ind]==numberToLooFor)- m)^2) +(((flatSegm[ind]==numberToLooFor)- m )^2) 
#         ssb += ((m - grandMean[1])^2)
#         end#if 
#     end#for
#     #now we accumulated ssw and ssb - we need to reduce it
#     offsetIter = UInt8(1)
#     while(offsetIter <32) 
#         @inbounds ssw+=shfl_down_sync(FULL_MASK, ssw, offsetIter)  
#         @inbounds ssb+=shfl_down_sync(FULL_MASK, ssb, offsetIter)  
#         offsetIter<<= 1
#     end
#         # now we have sums in first threads of the warp we need to pass it to shared memory
#         if(threadIdxX()==1)
#             @inbounds shmemSum[threadIdxY(),1]+=ssw
#             @inbounds shmemSum[threadIdxY(),2]+=ssb
#         end
#     sync_threads()
#     #finally reduce from shared memory
#         if(threadIdxY()==1)
#             offsetIter = UInt8(1)
#             while(offsetIter <32) 
#             @inbounds shmemSum[threadIdxX(),1]+=shfl_down_sync(FULL_MASK, shmemSum[threadIdxX(),1], offsetIter)  
#             @inbounds shmemSum[threadIdxX(),2]+=shfl_down_sync(FULL_MASK, shmemSum[threadIdxX(),2], offsetIter)  
#             offsetIter<<= 1
#             end
#     end  
#     sync_threads()sswTotal,ssbTotal
      #now in   shmemSum[1,1] we should have ssw and in  shmemSum[1,2] ssb
#       @ifXY 1 1  @inbounds @atomic sswTotal[]+= shmemSum[1,1]
#       @ifXY 1 2  @inbounds @atomic ssbTotal[]+= shmemSum[1,2]
#       @ifXY 1 3  begin
#         sswInner = shmemSum[1,2]/numberOfVoxels;
#         ssbInner = shmemSum[1,1]/(numberOfVoxels-1) * 2
#         @inbounds iccPerSlice[blockIdxX()]=(ssb - ssw)/(ssb + ssw)
#       end  
    # # ####### now we have ssw and ssb calculated both global and per slice

    return nothing



end




end#InterClassCorrKernel
