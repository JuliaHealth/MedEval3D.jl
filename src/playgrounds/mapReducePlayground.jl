

using Revise, Parameters, Logging

include("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
include("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\gpuUtils.jl")
using Main.BasicPreds, Main.GPUutils,Cthulhu,BenchmarkTools , CUDA

goldBoolGPU,segmBoolGPU,tp,tn,fp,fn, tpArr,tnArr,fpArr, fnArr, blockNum , nx,ny,nz ,tpTotalTrue,tnTotalTrue,fpTotalTrue, fnTotalTrue ,tpPerSliceTrue,  tnPerSliceTrue,fpPerSliceTrue,fnPerSliceTrue ,flattG, flattSeg ,FlattGoldGPU,FlattSegGPU,intermediateResTp,intermediateResFp,intermediateResFn = getSmallTestBools();


# Reduce a value across a warp
@inline function reduce_warp( vall, lanesNumb,i)
    offset = UInt32(1)
    while(offset <lanesNumb) 
        vall+=shfl_down_sync(FULL_MASK, vall, offset)  
        offset<<= 1
    end
    #CUDA.@cuprint "blockId+wid $(blockId+wid) maskTp  $(maskTp)  val  $(CUDA.popc(maskTp)[1]) \n"  
    # if(vall>2 && lanesNumb==32)
    #     CUDA.@cuprint "vall $(vall) lanesNumb $(lanesNumb)   i  $(i)\n"
    # end
    return vall
end





"""
add value to the shared memory in the position i, x where x is 1 ,2 or 3 and is calculated as described below
boolGold & boolSegm + boolGold +1 will evaluate to 
    3 in case  of true positive
    2 in case of false positive
    1 in case of false negative
"""
@inline function incr_shmem( primi::Int64,boolGold::Bool,boolSegm::Bool,shmem )
    # if boolGold & boolSegm  
    #      CUDA.@cuprint "primi $(primi)" 
    # end
    @inbounds shmem[ primi, (boolGold & boolSegm + boolSegm +1) ]+=(boolGold | boolSegm)
    return true
end



"""
adapted from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/src/mapreduce.jl
"""
function getBlockTpFpFn(goldBoolGPU::CuDeviceArray{Bool,1, 1}
        , segmBoolGPU::CuDeviceArray{Bool, 1, 1}
        ,tp,tn,fp,fn
        ,intermediateResTp::CuDeviceArray{Int32, 1, 1}
        ,intermediateResFp::CuDeviceArray{Int32, 1, 1}
        ,intermediateResFn::CuDeviceArray{Int32, 1, 1}
        ,numberOfDataPointsPerThread::Int64 =2)
    # we multiply thread id as we are covering now 2 places using one lane - hence after all lanes gone through we will cover 2 blocks - hence second multiply    
   # primi = (threadIdx().x) + ((blockIdx().x - 1) ) * (blockDim().x)# used as thread id
    i = (threadIdx().x* 2) + ((blockIdx().x - 1) *2) * (blockDim().x)# used as a basis to get data we want from global memory
    blockId = blockIdx().x
   wid, lane = fldmod1(threadIdx().x,32)

   #shared memory for  stroing intermidiate data per lane  
   shmem = @cuStaticSharedMem(Int32, (513,3))
   #for storing results from warp reductions
   shmemSumFn = @cuStaticSharedMem(Int32, (33))
   shmemSumFp = @cuStaticSharedMem(Int32, (33))
   shmemSumTp = @cuStaticSharedMem(Int32, (33))
  
    #incrementing - taking multiple datapoints per lane  
    incr_shmem(threadIdx().x,goldBoolGPU[i],segmBoolGPU[i],shmem)
    incr_shmem(threadIdx().x,goldBoolGPU[i+1],segmBoolGPU[i+1],shmem)
   #reducing across the warp
   sumFn = reduce_warp(shmem[threadIdx().x,1],32,i)
   sumFp = reduce_warp(shmem[threadIdx().x,2],32,i)
   sumTp = reduce_warp(shmem[threadIdx().x,3],32,i)



   #we are adding on separate threads results from warps to shared memory
    if(lane==1)
    #here we are reusing the shared memory
    if(sumFn>0)
       CUDA.@cuprint "sumFn   $(sumFn) wid $(wid) \n"
    end
        @inbounds shmemSumTp[wid]= sumTp
   # elseif(lane==3) 
        @inbounds shmemSumFp[wid]= sumFp
   # elseif(lane==4)
        @inbounds shmemSumFn[wid]= sumFn
    end#if  
sync_threads()
#now all data about of intrest should be in  shared memory so we will get all rsults from warp reduction in the shared memory
    # in case we have only 32 warps as we set we will not go out of bounds
      if(wid==2 )
        vallTp = reduce_warp(shmemSumTp[lane],32,i)
        #probably we do not need to sync warp as shfl dow do it for us        

        if(lane==1)
            @inbounds @atomic tp[]+=vallTp
            #@inbounds intermediateResTp[blockId]=vallTp
        end    
       elseif(wid==3 )   
        sync_warp()
        vallFp = reduce_warp(shmemSumFp[lane],32,i)
        if(lane==1)
            @inbounds @atomic fp[]+=vallFp
            #@inbounds intermediateResFp[blockId]=vallFp
        end    
       elseif(wid==5)  
        sync_warp()
        vallFn = reduce_warp(shmemSumFn[lane],32,i)
        if(lane==1)
            @inbounds @atomic fn[]+=vallFn
            #@inbounds intermediateResFn[blockId]=vallFn
        end  
        end

   return  
   end


#@benchmark CUDA.@sync 
#!!!!!!!!!!!!!!!! we need to reduce number of blocks depending on how many data points we access from single lane
#Int64(blockNum/2)    
# we are adding false in the end to make indexing easier
blockss = Int64(round((length(FlattGoldGPU)/512)/2))
@cuda threads=512 blocks=blockss getBlockTpFpFn(FlattGoldGPU,FlattSegGPU,tp,tn,fp,fn, intermediateResTp,intermediateResFp,intermediateResFn) 
tp[1]
tp[1] ==tpTotalTrue && fp[1] ==fpTotalTrue && fn[1] ==fnTotalTrue #tn[1] == tnTotalTrue && 

tp[1]
fp[1]
fn[1]

  @device_code_warntype interactive=true @cuda getBlockTpFpFn(FlattGoldGPU,FlattSegGPU,tp,tn,fp,fn, intermediateResTp,intermediateResFp,intermediateResFn)

## using occupancy API to calculate  threads number, block number etc...
args = (FlattGoldGPU,FlattSegGPU,tp,tn,fp,fn, intermediateResTp,intermediateResFp,intermediateResFn)
calcBlocks,valcThreads,maxBlocks = computeBlocksFromOccupancy(args,3 )




intermediateResTp,intermediateResFp,intermediateResFn
sum(intermediateResTp)
# from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/test/execution.jl
sum(FlattGoldGPU)
kernel = cufunction(kernelFunction, (FlattGoldGPU,FlattSegGPU,tp,tn,fp,fn, intermediateResTp,intermediateResFp,intermediateResFn))


x,y = fldmod1(68,32)






1024*1024*1024


function computeBlocksFromOccupancy(args, int32Shemm)
    wanted_threads =1000000
    function compute_threads(max_threads)
        if wanted_threads > max_threads
            true ? prevwarp(device(), max_threads) : max_threads
        else
            wanted_threads
        end
    end
    compute_shmem(threads) = Int64(threads*3*sizeof(Int16) )
    
       kernel = @cuda launch=false getBlockTpFpFn(args...) 
       kernel_config = launch_configuration(kernel.fun; shmem=compute_shmem∘compute_threads)
       blocks =  kernel_config.blocks
       threads =  kernel_config.threads
       maxBlocks = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    
return blocks,threads,maxBlocks
end


768/32








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









    
# # Reduce a value across a warp
# @inline function reduce_warp( vall, offset)
#     offset = UInt32(1)
#     while(offset <offset) 
#         vall+=shfl_down_sync(FULL_MASK, vall, offset)  
#         offset<<= 1
#     end
#     #CUDA.@cuprint "blockId+wid $(blockId+wid) maskTp  $(maskTp)  val  $(CUDA.popc(maskTp)[1]) \n"  
#     return vall
# end
# """
# add value to the shared memory in the position i, x where x is 1 ,2 or 3 and is calculated as described below
# boolGold & boolSegm + boolGold +1 will evaluate to 
#     3 in case  of true positive
#     2 in case of false positive
#     1 in case of false negative
# """
# @inline function incrementSmem!( i,boolGold,boolSegm )
#     shmem[ i,boolGold & boolSegm + boolGold +1 ]
# end

# """
# adapted from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/src/mapreduce.jl
# """
# function getBlockTpFpFn(goldBoolGPU::CuDeviceArray{Bool,1, 1}
#         , segmBoolGPU::CuDeviceArray{Bool, 1, 1}
#         ,tp,tn,fp,fn
#         ,intermediateResTp::CuDeviceArray{Int32, 1, 1}
#         ,intermediateResFp::CuDeviceArray{Int32, 1, 1}
#         ,intermediateResFn::CuDeviceArray{Int32, 1, 1}
#         ,numberOfDataPointsPerThread::Int64 =2)
#     # we multiply thread id as we are covering now 2 places using one lane - hence after all lanes gone through we will cover 2 blocks - hence second multiply    
#     primi = (threadIdx().x) + ((blockIdx().x - 1) ) * (blockDim().x)# used as thread id
#     i = (threadIdx().x* 2) + ((blockIdx().x - 1) *2) * (blockDim().x)# used as a basis to get data we want from global memory
#     blockId = blockIdx().x
#    wid, lane = fldmod1(threadIdx().x, warpsize())
#    #grid handle for cooperative groups
#    grid_handle = this_grid()

#    #shared memory for  stroing intermidiate data per lane and storing results from warp reductions
#    shmem = @cuStaticSharedMem(Int32, (768,3))
#    incrementSmem!(i,goldBoolGPU,segmBoolGPU)
#    incrementSmem!(i+1,goldBoolGPU,segmBoolGPU)

#    #getting the necessery data points to the given lane
#    datg1= goldBoolGPU[i] 
#    dats1= segmBoolGPU[i] 

#    datg2= goldBoolGPU[i+1] 
#    dats2= segmBoolGPU[i+1] 
#    #local to the lane data  accumulating number of true positives false positives ...
 
   

#    tpLoc::Int32 = (datg1 & dats1) + (datg2 & dats2)
#    fpLoc::Int32 = (~datg1 & dats1) + (~datg2 & dats2)
#    fnLoc::Int32 = (datg1 & ~dats1) + (datg2 & ~dats2)
   

#    sumTp = reduce_warp(tpLoc,32)
#    sumFp = reduce_warp(fpLoc,32)
#    sumFn = reduce_warp(fnLoc,32)

#    #we are adding on separate threads results from warps to shared memory
#     if(lane==1)


#         @inbounds  shmemTp[wid]= sumTp
#         @inbounds shmemFp[wid]+= sumFp
#         @inbounds shmemFn[wid]+= sumFn

#     # elseif(lane==2) 
#     #     @inbounds shmemFp[wid]+= reduce_warp(fpLoc)
#     # elseif(lane==3)
#     #      @inbounds shmemFn[wid]+= reduce_warp(fnLoc)
#     end#if  

# #now all data about of intrest should be in  shared memory so we will get all rsults from warp reduction in the shared memory
# sync_threads()
#     # in case we have only 32 warps as we set we will not go out of bounds
#       if(wid==1 )
#         vallTp = reduce_warp(shmemTp[lane],16)
#         #probably we do not need to sync warp as shfl dow do it for us        
#         if(lane==1)
#             @atomic tp[]+=vallTp
#             #@inbounds intermediateResTp[blockId]=vallTp
#         end    
#        elseif(wid==3 )   
#         vallFp = reduce_warp(shmemFp[lane],16)
#         if(lane==1)
#             @atomic fp[]+=vallFp
#             #@inbounds intermediateResFp[blockId]=vallFp
#         end    
#        elseif(wid==5)  
#         vallFn = reduce_warp(shmemFn[lane],16)
#         if(lane==1)
#             @atomic fn[]+=vallFn
#             #@inbounds intermediateResFn[blockId]=vallFn
#         end  
#         end
#    #synchronizing all of the device  
#    #sync_grid(grid_handle)      
#    #as we have now all of the device synchronized we can do the final stage of our reduction using intermediateRes

#    return  
#    end