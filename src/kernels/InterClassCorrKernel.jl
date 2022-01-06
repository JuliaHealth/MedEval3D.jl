"""

new optimazation idea  - try to put all data in boolean arrays in shared memory  when getting means
next we would need only to read shared memory - yet first one need to check wheather there would be enough shmem on device



calculating intercalss correlation
"""
module InterClassCorrKernel

using CUDA, Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils

export prepareInterClassCorrKernel,calculateInterclassCorr

"""
preparation for the interclass correlation kernel - we prepare the  GPU arrays - in this kernel they are small , calculate required loop iterations 
and using occupancy API calculate optimal number of  threads and blocks for both kernels 
also return arguments lists for both kernels
"""
function prepareInterClassCorrKernel()
  mainArrDims= (2,2,2)
  sumOfGold= CUDA.zeros(1);
  sumOfSegm= CUDA.zeros(1);
  sswTotal= CUDA.zeros(1);
  ssbTotal= CUDA.zeros(1);
  numberToLooFor= 1
  totalNumbOfVoxels= (mainArrDims[1]*mainArrDims[2]*mainArrDims[3])
  pixPerSlice = mainArrDims[1]*mainArrDims[2]
  iterLoop=5
  argsMain=(sumOfGold,sumOfSegm
     ,sswTotal
     ,ssbTotal
     ,iterLoop,pixPerSlice,totalNumbOfVoxels
     ,numberToLooFor )
  get_shmem(threads) = 4*33+3  #the same for both kernels
  threads,blocks =getThreadsAndBlocksNumbForKernel(get_shmem, kernel_InterClassCorr,(CUDA.zeros(2,2,2),CUDA.zeros(2,2,2),argsMain...))
  totalNumbOfVoxels= (mainArrDims[1]*mainArrDims[2]*mainArrDims[3])
  pixPerSlice= cld(totalNumbOfVoxels,blocks)
  iterLoop = UInt32(fld(pixPerSlice, threads[1]*threads[2]))

  argsMain=(sumOfGold,sumOfSegm
     ,sswTotal
     ,ssbTotal
     ,iterLoop,pixPerSlice,totalNumbOfVoxels
     ,numberToLooFor )


  return(argsMain, threads,blocks, totalNumbOfVoxels)
end









"""
calculates slicewise and global interclass correlation metric
"""
function calculateInterclassCorr(flatGold
                                ,flatSegm,threads,blocks
                                ,args,numberToLooFor)::Float64
 mainArrDims= size(flatGold)
 totalNumbOfVoxels= length(flatGold)
 pixPerSlice= cld(totalNumbOfVoxels,blocks)
 iterLoop = UInt32(fld(pixPerSlice, threads[1]*threads[2]))
 #resetting
 for i in  1:4   
  CUDA.fill!(args[i],0)
 end
 argsMain=(args[1],args[2]
 ,args[3]
 ,args[4]
 ,iterLoop,pixPerSlice,totalNumbOfVoxels
 ,numberToLooFor )           

@cuda threads=threads blocks=blocks cooperative = true kernel_InterClassCorr(flatGold,flatSegm,argsMain...)

  # println("grandMean[1] $(grandMean[1])  \n")
# @cuda threads=threads blocks=blocks  kernel_InterClassCorr(flatGold  ,flatSegm,args... )
     ssw = args[3][1]/args[7];
     ssb = args[4][1]/(args[7]-1) * 2;
    
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

function kernel_InterClassCorr(flatGold  ,flatSegm,sumOfGold,sumOfSegm
       ,sswTotal
       ,ssbTotal
       ,iterLoop,pixPerSlice,totalNumbOfVoxels
       ,numberToLooFor )
   grid_handle = this_grid()

   #for storing results from warp reductions
   shmemSum = @cuStaticSharedMem(Float32, (32,2))   #thread local values that are meant to store some results - like means ... 
   grandMeanShmem = @cuStaticSharedMem(Float32, (1))   
   offsetIter = UInt8(1)

   locValA = Float32(0)
   locValB = Float32(0)
        #reset shared memory
        @ifY 1 shmemSum[threadIdxX(),1]=0 ;   @ifY 2 shmemSum[threadIdxX(),2]=0
        sync_threads()

        #first we add 1 for each spot we have true - so we will get sum  - and from sum we can get mean
        @iterateLinearlyMultipleBlocks(iterLoop,pixPerSlice,totalNumbOfVoxels,begin
            locValA += flatGold[i]==numberToLooFor  
            locValB += flatSegm[i]==numberToLooFor
        end)#for
        @redWitAct(offsetIter,shmemSum,  locValA,+,     locValB,+  )
   sync_threads()
    @addAtomic(shmemSum, sumOfGold,sumOfSegm)
    ### sums should be in place
sync_grid(grid_handle)
grandMeanShmem[1]= ( (sumOfGold[1]/totalNumbOfVoxels) + (sumOfSegm[1]/totalNumbOfVoxels ))/2
locValA = 0
locValB = 0
@ifY 1 shmemSum[threadIdxX(),1]=0 ;   
@ifY 2 shmemSum[threadIdxX(),2]=0 ;
sync_threads()

@iterateLinearlyMultipleBlocks(iterLoop,pixPerSlice,totalNumbOfVoxels,begin
m =  ((flatGold[i]==numberToLooFor) +(flatSegm[i]==numberToLooFor))/2  
locValA += (((flatGold[i]==numberToLooFor)- m)^2) +(((flatSegm[i]==numberToLooFor)- m )^2) 
locValB += ((m - grandMeanShmem[1])^2)
end)#for
#now we accumulated ssw and locValB - we need to reduce it
offsetIter = UInt8(1)
@redWitAct(offsetIter,shmemSum,  locValA,+,     locValB,+  )
sync_threads()
@addAtomic(shmemSum, sswTotal,ssbTotal)
return nothing

end


# function kernel_InterClassCorr(flatGold
#     ,flatSegm,
#     sumOfGold,sumOfSegm
#        ,sswTotal
#        ,ssbTotal
#        ,iterLoop,pixPerSlice,totalNumbOfVoxels
#        ,numberToLooFor,grandMean )
  
#     #for storing results from warp reductions
#     shmemSum = @cuStaticSharedMem(Float32, (33,2))   #thread local values that are meant to store some results - like means ... 
#     @ifY 1 shmemSum[threadIdxX(),1]=0 ;   
#     @ifY 2 shmemSum[threadIdxX(),2]=0 ;
#     sync_threads()
#     locValA = Float32(0.0)
#     locValB = Float32(0.0)

#     @iterateLinearlyMultipleBlocks(iterLoop,pixPerSlice,totalNumbOfVoxels,begin
#         m =  ((flatGold[i]==numberToLooFor) +(flatSegm[i]==numberToLooFor))/2  
#         locValA += (((flatGold[i]==numberToLooFor)- m)^2) +(((flatSegm[i]==numberToLooFor)- m )^2) 
#         locValB += ((m - grandMean[1])^2)
#     end)#for
#     #now we accumulated ssw and locValB - we need to reduce it
#     offsetIter = UInt8(1)
#     @redWitAct(offsetIter,shmemSum,  locValA,+,     locValB,+  )
#    sync_threads()
#     @addAtomic(shmemSum, sswTotal,ssbTotal)
#     return nothing


# end





# function kernel_InterClassCorr_means(goldlGPU,segmGPU 
#      ,sumOfGold,sumOfSegm
#      ,sswTotal
#      ,ssbTotal
#      ,iterLoop,pixPerSlice,totalNumbOfVoxels
#      ,numberToLooFor )

# locValA = UInt32(0)
# locValB = UInt32(0)
# offsetIter= UInt8(1)
# shmemSum = @cuStaticSharedMem(UInt16, (33,2))   #thread local values that are meant to store some results - like means ... 
# @ifY 1 shmemSum[threadIdxX(),1]= 0
# @ifY 2 shmemSum[threadIdxX(),2]= 0
# sync_threads()

# @iterateLinearlyMultipleBlocks(iterLoop,pixPerSlice,totalNumbOfVoxels,
# #inner expression
# begin
# #updating variables needed to calculate means
# locValA += goldlGPU[i]==numberToLooFor  
# locValB += segmGPU[i]==numberToLooFor
# end) 

# #tell what variables are to be reduced and by what operation
# @redWitAct(offsetIter,shmemSum,  locValA,+,     locValB,+  )
# sync_threads()
# @addAtomic(shmemSum, sumOfGold, sumOfSegm)

# return nothing
# end



# function kernel_InterClassCorr(goldlGPU,segmGPU 
#   ,sumOfGold,sumOfSegm
#   ,sswTotal
#   ,ssbTotal
#   ,iterLoop,pixPerSlice,totalNumbOfVoxels
#   ,numberToLooFor )
# grandMean = @cuStaticSharedMem(Float32, (1)) 
# @ifXY 1 3 grandMean[1]= ( (sumOfGold[1]/totalNumbOfVoxels) + (sumOfSegm[1]/totalNumbOfVoxels ))/2
# ssw = Float32(0)
# ssb = Float32(0)
# m = Float32(0)
# offsetIter= UInt8(1)
# shmemSum = @cuStaticSharedMem(Float32, (33,2))   #thread local values that are meant to store some results - like means ... 
# @ifY 1 shmemSum[threadIdxX(),1]= 0
# @ifY 2 shmemSum[threadIdxX(),2]= 0
# sync_threads()
# @iterateLinearlyMultipleBlocks(iterLoop,pixPerSlice,totalNumbOfVoxels,
#  #inner expression
#  begin
#   m =  ((goldlGPU[i]==numberToLooFor) +(segmGPU[i]==numberToLooFor))/2  
#   ssw += (((goldlGPU[i]==numberToLooFor)- m)^2) +(((segmGPU[i]==numberToLooFor)- m )^2) 
#   ssb += ((m - grandMean[1])^2)
#  end) 

#  #tell what variables are to be reduced and by what operation
#  @redWitAct(offsetIter,shmemSum,  ssw,+,     ssb,+  )
# sync_threads()
# @addAtomic(shmemSum, sswTotal,ssbTotal)


#  return nothing



# end


# """
# calculates slicewise and global interclass correlation metric

# threadsForMeans, threadsForMain - tuples indicating dimensionality  of thread block 
# bsically all needed arguments apart from gold standard and algorithm 3 dm arrays  and numberToLooFor should be given by prepareInterClassCorrKernel
# """
# function calculateInterclassCorr(goldGPU,segmGPU,argsMain, threadsMain,  blocksMain,threadsMean,blocksMean,argsMean, totalNumbOfVoxels)::Float64

 
#   ## resetting some entries to 0 
# for i in 1:4
#   CUDA.fill!(argsMain[i],0)
# end
  
# @cuda threads=threadsMean  blocks=blocksMean  kernel_InterClassCorr_means(goldGPU,segmGPU,argsMean...)
# @cuda threads=threadsMain  blocks=blocksMain  kernel_InterClassCorr(goldGPU,segmGPU,argsMain...)

#   print("before correction ssw $(argsMain[4][1]) ssb $(argsMain[5][1])")

#      ssw = argsMain[4][1]/totalNumbOfVoxels;
#      ssb = (argsMain[5][1]/(totalNumbOfVoxels-1)) * 2;
#     print("ssw $(ssw) ssb $(ssb)")
# #  return (ssb - ssw)/(ssb + ssw);
#  return (ssb - ssw)/(ssb + ssw);
 
# end


# function kernel_InterClassCorr(goldlGPU,segmGPU 
#      ,sumOfGold,sumOfSegm
#      ,sswTotal
#      ,ssbTotal
#      ,iterLoop,pixPerSlice,totalNumbOfVoxels
#      ,numberToLooFor )

# grid_handle = this_grid()

#  locValA = Float32(0)
#  locValB = Float32(0)
#   offsetIter= 1
#   grandMean = @cuStaticSharedMem(Float32, (1)) 
#   grandMeanSq = @cuStaticSharedMem(Float32, (1)) 
#   halfMingrandSq = @cuStaticSharedMem(Float32, (1)) 
#   oneMingrandSq = @cuStaticSharedMem(Float32, (1)) 

#   shmemSum = @cuStaticSharedMem(Float32, (33,2))   #thread local values that are meant to store some results - like means ... 
#   @ifY 1 shmemSum[threadIdxX(),1]= Float32(0)
#   @ifY 2 shmemSum[threadIdxX(),2]= Float32(0)

#   sync_threads()
#   @iterateLinearlyMultipleBlocks(iterLoop,pixPerSlice,totalNumbOfVoxels,
#     #inner expression
#     begin
#         #updating variables needed to calculate means
#            locValA += goldlGPU[i]==numberToLooFor  
#            locValB += segmGPU[i]==numberToLooFor
#     end) 

#     #tell what variables are to be reduced and by what operation
#     @redWitAct(offsetIter,shmemSum,  locValA,+,     locValB,+  )
#   sync_threads()
#   @addAtomic(shmemSum, sumOfGold, sumOfSegm)

#   # @ifX 1  if(threadIdxY()>22 ) CUDA.@cuprint "aaa idY $(threadIdxY()) \n" end

#  sync_grid(grid_handle)
# #by now we should have all means 
#  @ifXY 1 3 grandMean[1]= ( (sumOfGold[1]/totalNumbOfVoxels) + (sumOfSegm[1]/totalNumbOfVoxels ))/2
#  @ifXY 2 3 grandMeanSq[1]= ((( (sumOfGold[1]/totalNumbOfVoxels) + (sumOfSegm[1]/totalNumbOfVoxels ))/2))^2
#  @ifXY 3 3 halfMingrandSq[1]= (0.5- (( (sumOfGold[1]/totalNumbOfVoxels) + (sumOfSegm[1]/totalNumbOfVoxels ))/2))^2
#  @ifXY 4 3 oneMingrandSq[1]= (1- (( (sumOfGold[1]/totalNumbOfVoxels) + (sumOfSegm[1]/totalNumbOfVoxels ))/2))^2

#  @ifXY 1 1 CUDA.@cuprint "  "

#  @ifY 1 shmemSum[threadIdxX(),1]= Float32(0)
#  @ifY 2 shmemSum[threadIdxX(),2]= Float32(0)
#  sync_threads()
#  locValA = 0
#  locValB = 0
#   # m = Float32(0)
# sync_threads()

#   @iterateLinearlyMultipleBlocks(iterLoop,pixPerSlice,totalNumbOfVoxels,
#     begin
#       boolGold = segmGPU[i]==numberToLooFor
#       boolSegm = goldlGPU[i]==numberToLooFor
#       if(boolGold && boolSegm)
#         # CUDA.@cuprint "aa    "
#         locValB = locValB+oneMingrandSq[1] 
#       elseif(boolGold ⊻ boolSegm)
#        locValA =locValA+ (0.25)
#        locValB += halfMingrandSq[1]#((0.5 - grandMean[1])*(0.5 - grandMean[1])   )
#       else
#         locValB =locValB+ grandMeanSq[1]         
#       end  

#       # elseif(boolGold || boolSegm)
#       #   m = 0.5#((boolGold ) +(boolSegm) )/2 
#       #   locValA += 0.25
#       #   locValB += ((0.5 - grandMean[1])^2)
#       # else
#       #   locValB += grandMean[1]^2

#       # end   
#     end ) 
#     #tell what variables are to be reduced and by what operation
#       offsetIter= 1
#       @redWitAct(offsetIter,shmemSum,  locValA,+,     locValB,+  )
#    sync_threads()
#     @addAtomic(shmemSum, sswTotal,ssbTotal)
#     #         #now we will have sum of all entries in given slice that comply to our predicate
#     #         #next we need to reduce values
#     #         while(offsetIter <32) 
#     #             @inbounds locValA+=shfl_down_sync(FULL_MASK, locValA, offsetIter)  
#     #             @inbounds locValB+=shfl_down_sync(FULL_MASK, locValB, offsetIter)  
#     #             offsetIter<<= 1
#     #         end
#     #         # @ifX 1  if(threadIdxY()>22 ) CUDA.@cuprint "bbb idY $(threadIdxY()) \n" end
#     #         #@ifX 1  if(blockIdxX()==1 ) CUDA.@cuprint "bbb idY $(threadIdxY()) \n" end
#     #         #@ifY 1  if(blockIdxX()==1 ) CUDA.@cuprint "bbb idX $(threadIdxX()) \n" end

#     #         # @ifX 1  if(blockIdxX()==1 ) CUDA.@cuprint "bbb idY $(threadIdxY()) \n" end
#     #         #@ifY 1  if(blockIdxX()==1 ) CUDA.@cuprint "bbb idX $(threadIdxX()) \n" end

#     #  #   now we have sums in first threads of the warp we need to pass it to shared memory
#     #     if(threadIdxX()==1)
#     #       #TODO() try to do full reduction without those atomics 
#     #         @inbounds @atomic sswTotal[1]+=locValA
#     #         @inbounds @atomic ssbTotal[1]+=locValB
#     #     end
#         # sync_threads()

#         # sync_warp(active_mask())

#         # #finally reduce from shared memory
#         # if(threadIdxY()==1)
#         #     offsetIter = UInt8(1)
#         #     while(offsetIter <32) 
#         #         @inbounds shmemSum[threadIdxX(),1]+=shfl_down_sync(FULL_MASK, shmemSum[threadIdxX(),1], offsetIter)  
#         #         @inbounds shmemSum[threadIdxX(),2]+=shfl_down_sync(FULL_MASK, shmemSum[threadIdxX(),2], offsetIter)  
#         #     offsetIter<<= 1
#         #     end
#         # end  



# # @ifXY 1 1 CUDA.@cuprint "shmemSum[1,1]  $(shmemSum[1,1])  \n "
#   #  sync_threads()
#   #  @addAtomic(shmemSum, sswTotal,ssbTotal)
  

#     return nothing



# end




end#InterClassCorrKernel







  
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
#         sswInner = shmemSum[1,2]/totalNumbOfVoxels;
#         ssbInner = shmemSum[1,1]/(totalNumbOfVoxels-1) * 2
#         @inbounds iccPerSlice[blockIdxX()]=(ssb - ssw)/(ssb + ssw)
#       end  
    # # ####### now we have ssw and ssb calculated both global and per slice














# """
# adapted from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/src/mapreduce.jl
# goldGPU - array holding data of gold standard  array
# segmGPU -  array with the data we want to compare with gold standard
# ic - holding single value for global interclass correlation
# intermediateRes- array holding slice wise results for ic
# loopXdim,loopYdim,loopZdim - number of times we need to loop over each dimension in order to get all needed data
# sliceEdgeLength - length of edge of the slice we need to square this number to get number of pixels in a slice
# amountOfWarps - how many warps we can stick in the block
# """

# function kernel_InterClassCorr_means(goldlGPU
#                                ,segmGPU
#                                 ,iterLoop,pixPerSlice,totalNumbOfVoxels
#                                 ,sumOfGold
#                                 ,sumOfSegm
#                                 ,numberToLooFor )
  
#   locValA = UInt32(0)
#  locValB = UInt32(0)
#   offsetIter= UInt8(1)
#   shmemSum = @cuStaticSharedMem(UInt16, (33,2))   #thread local values that are meant to store some results - like means ... 
#   @iterateLinearlyMultipleBlocks(iterLoop,pixPerSlice,totalNumbOfVoxels,
#     #inner expression
#     begin
#         #updating variables needed to calculate means
#            locValA += goldlGPU[i]==numberToLooFor  
#            locValB += segmGPU[i]==numberToLooFor
#     end) 

#     #tell what variables are to be reduced and by what operation
#     @redWitAct(offsetIter,shmemSum,  locValA,+,     locValB,+  )
#   sync_threads()
#   @addAtomic(shmemSum, sumOfGold, sumOfSegm)

# #     #offset for lloking for values in source arrays 
# #     offset = (pixelNumberPerSlice*(blockIdx().x-1))
# #    #for storing results from warp reductions
# #    shmemSum = @cuStaticSharedMem(UInt16, (33,2))   #thread local values that are meant to store some results - like means ... 
# #    offsetIter = UInt8(1)

# #    locValA = UInt32(0)
# #    locValB = UInt32(0)
# #         #reset shared memory
# #         @ifY 1 shmemSum[threadIdxX(),1]=0 ;   @ifY 2 shmemSum[threadIdxX(),2]=0
# #         sync_threads()

# #         #first we add 1 for each spot we have true - so we will get sum  - and from sum we can get mean
# #         @unroll for k in 0:loopNumb
# #             if(threadIdxX()+(threadIdxY()-1)*32+k*1024 <=pixelNumberPerSlice)
# #             ind =offset+ threadIdxX()+(threadIdxY()-1)*32+k*1024
# #             locValA += flatGold[ind]==numberToLooFor  
# #             locValB += flatSegm[ind]==numberToLooFor
# #             end#if 
# #         end#for
        

# #             #now we will have sum of all entries in given slice that comply to our predicate
# #             #next we need to reduce values
# #             offsetIter = UInt8(1)
# #             while(offsetIter <32) 
# #                 @inbounds locValA+=shfl_down_sync(FULL_MASK, locValA, offsetIter)  
# #                 @inbounds locValB+=shfl_down_sync(FULL_MASK, locValB, offsetIter)  
# #                 offsetIter<<= 1
# #             end
# #         # now we have sums in first threads of the warp we need to pass it to shared memory
# #         if(threadIdxX()==1)

# #             @inbounds shmemSum[threadIdxY(),1]+=locValA
# #             @inbounds shmemSum[threadIdxY(),2]+=locValB
# #         end
# #         sync_threads()
# #         #finally reduce from shared memory
# #         if(threadIdxY()==1)
# #             offsetIter = UInt8(1)
# #             while(offsetIter <32) 
# #                 @inbounds shmemSum[threadIdxX(),1]+=shfl_down_sync(FULL_MASK, shmemSum[threadIdxX(),1], offsetIter)  
# #                 @inbounds shmemSum[threadIdxX(),2]+=shfl_down_sync(FULL_MASK, shmemSum[threadIdxX(),2], offsetIter)  
# #             offsetIter<<= 1
# #             end
# #         end  
# # sync_threads()
       
# #       #now in   shmemSum[1,1] we should have sum of values complying with our predicate in gold mask and in shmemSum[1,2] values of other mask
# #       #we need to add now those to the globals  
# #       @ifXY 1 1  @inbounds @atomic sumOfGold[]+= shmemSum[1,1]
# #       @ifXY 1 2  @inbounds @atomic sumOfSegm[]+= shmemSum[1,2]

#     return nothing
# end

