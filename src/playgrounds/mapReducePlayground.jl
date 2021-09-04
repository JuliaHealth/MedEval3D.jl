

using Revise, Parameters, Logging

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\gpuUtils.jl")
using Main.BasicPreds, Main.GPUutils,Cthulhu,BenchmarkTools , CUDA

goldBoolGPU,segmBoolGPU,tpTnFpFn, tpArr,tnArr,fpArr, fnArr, blockNum , nx,ny,nz,xthreads, ythreads,zthreads ,tpTotalTrue,tnTotalTrue,fpTotalTrue, fnTotalTrue ,tpPerSliceTrue,  tnPerSliceTrue,fpPerSliceTrue,fnPerSliceTrue ,flattG, flattSeg ,FlattGoldGPU,FlattSegGPU = getSmallTestBools();

FlattGoldGPU,FlattSegGPU ;

# Reduce a value across a warp
@inline function reduce_warp(op, val,arr1,arr2,index)
    offset = UInt32(1)
    while offset < warpsize()
        shuffled= shfl_down_sync(FULL_MASK, arr1[i], offset)
        if shuffled!=false
        CUDA.@cuprint "shuffled $(shuffled) "
        end
        val = op(val,shfl_down_sync(FULL_MASK, arr1[i], offset))
        offset <<= 1
    end

    return val
end

"""
adapted from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/src/mapreduce.jl
"""
function kernelFunction(goldBoolGPU::CuDeviceArray{Bool,1, 1}, segmBoolGPU::CuDeviceArray{Bool, 1, 1},tpTnFpFn,warpsInBlock::Int64)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
   wid, lane = fldmod1(threadIdx().x, warpsize()) 
   
    #shared memory
   shmem = @cuStaticSharedMem(Float32, (256))     #warpsInBlock !! make it dynamic - makro?
   #using native function we calculate how many threads pass our criteria 
   maskTp = vote_ballot_sync(FULL_MASK,goldBoolGPU[i] & segmBoolGPU[i])  
   maskFp = vote_ballot_sync(FULL_MASK,~goldBoolGPU[i] & segmBoolGPU[i])  
   maskFn = vote_ballot_sync(FULL_MASK,goldBoolGPU[i] & ~segmBoolGPU[i])  
   # generally values for  maskTp, maskFp, maskFn are constant across the warp  so in order to prevent adding the same number couple times we need modulo operator
   #modul = threadIdx().x % 32
   
   if(lane==1)
    @atomic tpTnFpFn[1]+= CUDA.popc(maskTp)[1] *1
#    CUDA.@cuprint "maskTp  $(maskTp) \n"  
   val = 0  
#    offset = UInt32(1) 

#    while offset < warpsize()
#     val+= shfl_down_sync(maskTp,1,offset)  
#     offset <<= 1
#     end#while
#  @atomic tp[]+=CUDA.popc(maskTp)[1]


    CUDA.@cuprint "maskTp  $(maskTp)  val  $(CUDA.popc(maskTp)[1]) \n"  

end#if  

   return  

    end
    warpsInBlock = Int64((xthreads*ythreads*zthreads )/32)

    @cuda threads=xthreads*ythreads*zthreads blocks=blockNum kernelFunction(FlattGoldGPU,FlattSegGPU,tpTnFpFn, warpsInBlock) #shmem= 10*10*10*2  threads=(8,8,8)   blocks=ceil(Int,n/8*8*8)

tpTnFpFn[:]

valc = 96


  @device_code_warntype interactive=true @cuda kernelFunction(FlattGoldGPU,FlattSegGPU,tpTnFpFn, warpsInBlock)


























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