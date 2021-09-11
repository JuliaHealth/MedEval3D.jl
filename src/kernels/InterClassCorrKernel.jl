"""
calculating intercalss correlation
"""
module InterClassCorrKernel

using Main.BasicPreds, Main.GPUutils, CUDA
using Main.MainOverlap, Main.TpfpfnKernel

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

function kernel_InterClassCorr(flatGold
                               ,flatSegm
                                ,loopNumb::Int64
                                ,indexCorr::Int64
                                ,slicePixelNumb::Int64
                                ,totalVoxelNumb::Int64
                                ,sumOfGold
                                ,sumOfSegm
                                ,sumOfGoldPartials
                                ,sumOfSegmPartials
                                ,sswPartials
                                ,sswTotal
                                ,ssbPartialsperSlice
                                ,SsbTotalGlobal
                                ,slicesPerBlockMatrix
                                ,maxSlicesPerBlock::Int64)
   wid, lane = fldmod1(threadIdx().x,32)
   grid_handle = this_grid()

      #shared memory for  stroing intermidiate data per lane  
      shmem = @cuStaticSharedMem(Float16, (513,3))
      #for storing results from warp reductions
      shmemSum = @cuStaticSharedMem(Float32, (33,3))
 ####### first we need to get the global mean of all images and of each slice  
    for blockRef in 1:maxSlicesPerBlock
        sliceNumb= blockIdx().x#slicesPerBlockMatrix[blockIdx().x,blockRef]
        if(sliceNumb>-1)
            i = (threadIdx().x* indexCorr) + (sliceNumb) * (blockDim().x)# used as a basis to get data we want from global memory
            # at the begining of analyzing each slice we need to set shared memory to 0 
            shmem[threadIdx().x, 1]=0
            shmem[threadIdx().x, 2]=0
            shmemSum[wid,1]=0
            shmemSum[wid,2]=0
            # simple adding to the shared memory in order to  be able to calculate mean later
            @unroll for k in 0:loopNumb
                incr_shmem_forMean(threadIdx().x,flatGold[i+k],flatSegm[i+k],shmem)
            end#for 
            #reducing across the warp
            firstReduce(shmem,shmemSum,wid,threadIdx().x,lane)
            sync_threads()
            getSecondBlockReduce( 1,1,wid,sumOfGoldPartials,sumOfGoldTotal,shmemSum,sliceNumb,lane)
            getSecondBlockReduce( 2,2,wid,sumOfSegmPartials,sumOfSegmTotal,shmemSum,sliceNumb,lane)
        end#if    
    end#for
    # sync_grid(grid_handle)
    # ######now sum of elements both  slice wise and global should be available and would make it easy to calculate mean
    # for blockRef in 1:maxSlicesPerBlock
    #     sliceNumb= slicesPerBlockMatrix[blockIdx().x,blockRef]
    #     if(sliceNumb>-1)
    #         i = (threadIdx().x* indexCorr) + (sliceNumb) * (blockDim().x)# used as a basis to get data we want from global memory
    #         # at the begining of analyzing each slice we need to set shared memory to 0 
    #         shmem[threadIdx().x, 1]=0
    #         shmem[threadIdx().x, 2]=0
    #         shmem[threadIdx().x, 3]=0
    #         shmemSum[wid,1]=0
    #         shmemSum[wid,2]=0
    #         shmemSum[wid,3]=0
            
    #         #loading  slice and global mean
    #         if(threadIdx().x==1)
    #         shmemSum[33,1]=  (sumOfGoldPartials[sliceNumb] +sumOfSegmPartials[sliceNumb])/slicePixelNumb     #slice mean
    #         end
    #         if(threadIdx().x==1)
    #         shmemSum[33,2]= (sumOfGold[1]+sumOfSegm[1])/totalVoxelNumb #global mean   
    #         end
    #         sync_threads()

    #         # simple adding to the shared memory in order to  be able to calculate mean later
    #         @unroll for k in 0:loopNumb
    #             incr_shmem_forRes(threadIdx().x,flatGold[i+k],flatSegm[i+k],shmem,shmemSum[33,1],shmemSum[33,2] )
    #         end#for 
    #         #reducing across the warp
    #         firstReduce(shmem,shmemSum,wid,threadIdx().x,lane)
    #         sync_threads()
    #         getSecondBlockReduce( 1,1,wid,sswPartials,sswTotal,shmemSum,sliceNumb,lane)
    #         getSecondBlockReduceNoAtomic( 2,2,wid,ssbPartialsperSlice,shmemSum,sliceNumb,lane)
    #         getSecondBlockReduceNoPerSlice( 3,3,wid,SsbTotalGlobal,shmemSum,sliceNumb,lane)
        
    #     end#if    
    # end#for

    # ####### now we have ssw and ssb calculated both global and per slice
    # sync_grid(grid_handle)

    # ssw = ssw/numberElements 
    # ssb = ssb/(numberElements-1) * 2 
    # icc = (ssb - ssw)/(ssb + ssw) 
    return nothing
end





"""
now plan is to create a matrix on the CPU that will be pushed onto the GPU
matrix will tell each block which slices it should manage in order to cover all slices
number of blocks will be calclulated in a way to fit them all in a single grid and enable full utilization of cooperative groups 
matrix will be n x m  where 
    n is number of blocks we can fit in the grid
    m is maximum number of slices managed per block - of course in most cases some blocks will not be needed in final round - in this case this will evaluate to 0 ...
  code adapted from 
    https://stackoverflow.com/questions/63929929/processing-shared-work-queue-using-cuda-atomic-operations-and-grid-synchronizati/63930239#63930239
    https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#grid-synchronization-cg
    https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/test/execution.jl

    generally the number of threads to avoid corner cases  needs to be 256 or 512 or 1024

    args - tuple with arguments needed to execute a kernel (it will not be executed it is only needed for occupancy API)
    kernelFun - the function describing kernel
    slicesNumb - number of slices to manage, generally it is assumed that each block controls one slice
    """
function assignWorkToCooperativeBlocks(args, kernelFun )
    kernel = @cuda launch=false kernelFun(args...) 
    return  CUDA.active_blocks(kernel.fun, 512)

end #assignWorkToCooperativeBlocks



"""
add value to the shared memory in the position i, x 
    where x is 1 for  gold standard mask and 2 for one we test
"""
@inline function incr_shmem_forMean( primi::Int64,goldVal,segmVal,shmem )
    @inbounds shmem[ primi, 1 ]+=goldVal
    @inbounds shmem[ primi, 2 ]+=segmVal
    return true
end


"""
accumulates values needed for calculating intercalss correlation
"""
@inline function incr_shmem_forRes( primi::Int64,goldVal,segmVal,shmem, slicemean, globalmean )
    m = (goldVal + segmVal)/2
    @inbounds shmem[ primi, 1 ]+=( (goldVal-m)^2 + (segmVal-m)^2 ) 
    @inbounds shmem[ primi, 2 ]+=(m- slicemean)^2
    @inbounds shmem[ primi, 3 ]+=(m- globalmean)^2
    return true
end



"""
reduction across the warp and adding to appropriate spots in the  shared memory
"""
function firstReduce(shmem,shmemSum,wid,threadIdx,lane   )
    @inbounds sumA = reduce_warp(shmem[threadIdx,1],32)
    @inbounds sumB = reduce_warp(shmem[threadIdx,2],32)
    @inbounds sumC = reduce_warp(shmem[threadIdx,3],32)

    if(lane==1)
   @inbounds shmemSum[wid,1]= sumA
     end  
    if(lane==2) 
       @inbounds shmemSum[wid,2]= sumB
    end     
    if(lane==3) 
        @inbounds shmemSum[wid,3]= sumC
     end     
end#firstReduce

"""
sets the final block amount to the  array representing each slice, 
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
  end  
end#getSecondBlockReduce
"""
modifaction without using atomics ...
"""
function getSecondBlockReduceNoAtomic(chosenWid,numb,wid, intermediateRes,shmemSum,blockId,lane)
    if(wid==chosenWid )
        shmemSum[33,numb] = reduce_warp(shmemSum[lane,numb],32 )
      if(lane==2)
        @inbounds intermediateRes[blockId]=shmemSum[33,numb]
      end    
  end  
end#getSecondBlockReduceNoAtomic
"""
modifaction without updating per slice data
"""
function getSecondBlockReduceNoPerSlice(chosenWid,numb,wid, singleREs,shmemSum,blockId,lane)
    if(wid==chosenWid )
        shmemSum[33,numb] = reduce_warp(shmemSum[lane,numb],32 )
        
      #probably we do not need to sync warp as shfl dow do it for us         
      if(lane==1)
          @inbounds @atomic singleREs[]+=shmemSum[33,numb]
      end    
  end  
end#getSecondBlockReduceNoPerSlice

end#InterClassCorrKernel