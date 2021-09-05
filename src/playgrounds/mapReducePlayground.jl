

using Revise, Parameters, Logging

include("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
include("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\gpuUtils.jl")
using Main.BasicPreds, Main.GPUutils,Cthulhu,BenchmarkTools , CUDA

goldBoolGPU,segmBoolGPU,tp,tn,fp,fn, tpArr,tnArr,fpArr, fnArr, blockNum , nx,ny,nz ,tpTotalTrue,tnTotalTrue,fpTotalTrue, fnTotalTrue ,tpPerSliceTrue,  tnPerSliceTrue,fpPerSliceTrue,fnPerSliceTrue ,flattG, flattSeg ,FlattGoldGPU,FlattSegGPU,intermediateResTp,intermediateResFp,intermediateResFn = getSmallTestBools();


# Reduce a value across a warp
@inline function reduce_warp( vall)

    offset = UInt32(1)
    while(offset <32) 
        vall+=shfl_down_sync(FULL_MASK, vall, offset)  
        offset<<= 1
    end

    #CUDA.@cuprint "blockId+wid $(blockId+wid) maskTp  $(maskTp)  val  $(CUDA.popc(maskTp)[1]) \n"  

    return vall
end

"""
adapted from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/src/mapreduce.jl
"""
function kernelFunction(goldBoolGPU::CuDeviceArray{Bool,1, 1}, segmBoolGPU::CuDeviceArray{Bool, 1, 1},tp,tn,fp,fn,intermediateResTp::CuDeviceArray{Int32, 1, 1},intermediateResFp::CuDeviceArray{Int32, 1, 1},intermediateResFn::CuDeviceArray{Int32, 1, 1} )
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    blockId = blockIdx().x
   wid, lane = fldmod1(threadIdx().x, warpsize())

   #shared memory for storing results from warp reductions
   shmemTp = @cuStaticSharedMem(Int32, (33))
   shmemFp = @cuStaticSharedMem(Int32, (33))
   shmemFn = @cuStaticSharedMem(Int32, (33))

   
   @inbounds goldb::Bool =goldBoolGPU[i]
   @inbounds segmb::Bool =segmBoolGPU[i] 
   #using native function we calculate how many threads pass our criteria 
   maskTp = vote_ballot_sync(FULL_MASK,goldb & segmb)  
   maskFp = vote_ballot_sync(FULL_MASK,~goldb & segmb)  
   maskFn = vote_ballot_sync(FULL_MASK,goldb & ~segmb)  
   
   #we are adding on separate threads results from warps to shared memory
    if(lane==1)
        @inbounds  shmemTp[wid]= CUDA.popc(maskTp)[1]*1
    elseif(lane==2) 
        @inbounds shmemFp[wid]+= CUDA.popc(maskFp)[1]*1
    elseif(lane==3)
         @inbounds shmemFn[wid]+= CUDA.popc(maskFn)[1]*1
    end#if  

#now all data about of intrest should be in  shared memory so we will get all rsults from warp reduction in the shared memory
sync_threads()
    # in case we have only 32 warps as we set we will not go out of bounds
      if(wid==1 )
        vallTp = reduce_warp(shmemTp[lane])
        #probably we do not need to sync warp as shfl dow do it for us        
        if(lane==1)
            @inbounds intermediateResTp[blockId]=vallTp
        end    
       elseif(wid==3 )   
        vallFp = reduce_warp(shmemFp[lane])
        if(lane==1)
            @inbounds intermediateResFp[blockId]=vallFp
        end    
       elseif(wid==5)  
        vallFn = reduce_warp(shmemFn[lane])
        if(lane==1)
            @inbounds intermediateResFn[blockId]=vallFn
        end  
        end

    #     # CUDA.@cuprint " blockId  $( blockId)  \n" 
    #     sync_threads()
    #     intermediateResults[blockId+1,1]=shmemA[1]
    # elseif(blockDim().x == threadIdx().x+2)
    #     sync_threads()
    #     intermediateResults[blockId+1,2]=shmemA[2]
    # elseif(blockDim().x == threadIdx().x+1) 
    #     sync_threads()        
    #     intermediateResults[blockId+1,3]=shmemA[3]


    #   CUDA.@cuprint "maskTp  $(maskTp)  val  $(CUDA.popc(maskTp)[1]) \n"  
    #   end  

   return  

    end


#@benchmark CUDA.@sync 
      @cuda threads=1024 blocks=blockNum kernelFunction(FlattGoldGPU,FlattSegGPU,tp,tn,fp,fn, intermediateResTp,intermediateResFp,intermediateResFn) 
      
      intermediateResTp,intermediateResFp,intermediateResFn
      sum(intermediateResFn)

maximum(intermediateResFn)

      tp[]
   

tp[1] ==tpTotalTrue && fp[1] ==fpTotalTrue && fn[1] ==fnTotalTrue #tn[1] == tnTotalTrue && 


  @device_code_warntype interactive=true @cuda kernelFunction(FlattGoldGPU,FlattSegGPU,tp,tn,fp,fn, warpsInBlock,intermediateResults)




x,y = fldmod1(68,32)





















for i in 1:32 
    xx = 224>>(i-1) & 1
    if(xx==1) @info 1   end  
end


    
# @benchmark CUDA.@sync  blockNum
@cuda threads=xthreads*ythreads*zthreads blocks=blockNum kernelFunction(FlattGoldGPU,FlattSegGPU,tp,tn,fp, fn, warpsInBlock) #shmem= 10*10*10*2  threads=(8,8,8)   blocks=ceil(Int,n/8*8*8)
#testing correctness

tp[1] ==tpTotalTrue

tn[1] == tnTotalTrue && tp[1] ==tpTotalTrue && fp[1] ==fpTotalTrue && fn[1] ==fnTotalTrue





nx*ny*nz - tpTotalTrue - fpTotalTrue -  fnTotalTrue

2100/256

2046-1790

tp[1]==2 && tn[1]==(nx*ny*nz)-6 && fp[1]==1 && fn[1]==3

(nx*ny*nz)/(xthreads*ythreads*zthreads)
#@cuda threads=(8,8,8) blocks=blockNum  kernelFunction(goldBoolGPU,segmBoolGPU,tp,tn,fp, fn) #shmem= 10*10*10*2  threads=(8,8,8)   blocks=ceil(Int,n/8*8*8)




using CUDA
goldBoolGPU= CuArray(falses(16,16,2));
segmBoolGPU= CuArray(falses(16,16,2));
fn = CuArray([0])
function kernelFunct(goldBoolGPU::CuDeviceArray{Bool, 3, 1}, segmBoolGPU::CuDeviceArray{Bool, 3, 1},fn)
    i= (blockIdx().x) * blockDim().x + threadIdx().x
    j = (blockIdx().y) * blockDim().y + threadIdx().y
    z = (blockIdx().z) * blockDim().z + threadIdx().z 

    if (goldBoolGPU[i] & !segmBoolGPU[i] )
        @atomic fn[]+=1    
        end
    return  
    end

    @device_code_warntype kernelFunct(goldBoolGPU,segmBoolGPU,fn) 

@cuda threads=(4, 4,1) blocks=32  kernelFunct(goldBoolGPU,segmBoolGPU,fn) 
#I get error ERROR: Out-of-bounds array access.


fn

(16*16*2)/(4*4)


128*128*2/16

    # kernel = @cuda launch=false kernelFunction(arr1, arr2, res)
    #config = launch_configuration(kernel.fun, shmem=threads-> 2 * sum(threads) * sizeof(Float32))
    




#z= view(arr1,:,:,1) # in such configuration data is contiguous  Base.iscontiguous(x)
#arr1=  CUDA.ones(10,128,128) ;  # 3 dim array of ones



    # CUDA.@cuprint "i $(i) ; (blockIdx().x-1) $(blockIdx().x-1) ; blockDim().x $(blockDim().x) ; threadIdx().x $(threadIdx().x)               \n "
    # CUDA.@cuprint "j $(j) ; blockIdx().y $(blockIdx().y-1) ; blockDim().y $(blockDim().y) ; threadIdx().y $(threadIdx().y)               \n"
    # CUDA.@cuprint "z $(z) ; blockIdx().z $(blockIdx().z-1) ; blockDim().z $(blockDim().z) ; threadIdx().z $(threadIdx().z)               \n"
    #CUDA.@cuprint "goldBoolGPU[i,j,z] $(goldBoolGPU[i,j,z]) segmBoolGPU[i,j,z] $(segmBoolGPU[i,j,z]) i $(i) j $(j) z $(z)  \n"



    # function kernelFunction(goldBoolGPU::CuDeviceArray{Bool, 3, 1}, segmBoolGPU::CuDeviceArray{Bool, 3, 1},tp,tn,fp, fn)
    #     # getting all required indexes
    #     i,j,z = defineIndicies()
    #         if(goldBoolGPU[i,j,z] && segmBoolGPU[i,j,z] )
    #         CUDA.@cuprint "  gold $(goldBoolGPU[i,j,z])  segm  $(segmBoolGPU[i,j,z]) tp + \n"
    #         @atomic tp[]+=1
    #         elseif (!goldBoolGPU[i,j,z] && !segmBoolGPU[i,j,z] )
    #         CUDA.@cuprint "  gold $(goldBoolGPU[i,j,z])  segm  $(segmBoolGPU[i,j,z]) tn +\n"
    #         @atomic tn[]+=1
    #         elseif (!goldBoolGPU[i,j,z] && segmBoolGPU[i,j,z] )
    #         CUDA.@cuprint " gold $(goldBoolGPU[i,j,z])  segm  $(segmBoolGPU[i,j,z])  fp +\n"      
    #         @atomic fp[]+=1    
    #         elseif (goldBoolGPU[i,j,z] && !segmBoolGPU[i,j,z] )
    #         CUDA.@cuprint " gold $(goldBoolGPU[i,j,z])  segm  $(segmBoolGPU[i,j,z]) fn +\n"       
    #         @atomic fn[]+=1    
    #         end
    #       return  
    #     end