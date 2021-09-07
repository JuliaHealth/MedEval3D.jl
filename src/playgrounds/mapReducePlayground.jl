

using Revise, Parameters, Logging

include("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
include("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\gpuUtils.jl")
using Main.BasicPreds, Main.GPUutils,Cthulhu,BenchmarkTools , CUDA

goldBoolGPU,segmBoolGPU,tp,tn,fp,fn, tpArr,tnArr,fpArr, fnArr, blockNum , nx,ny,nz ,tpTotalTrue,tnTotalTrue,fpTotalTrue, fnTotalTrue ,tpPerSliceTrue,  tnPerSliceTrue,fpPerSliceTrue,fnPerSliceTrue ,flattG, flattSeg ,FlattGoldGPU,FlattSegGPU,intermediateResTp,intermediateResFp,intermediateResFn = getSmallTestBools();


# Reduce a value across a warp
@inline function reduce_warp( vall, lanesNumb)
    offset = UInt32(1)
    while(offset <lanesNumb) 
        vall+=shfl_down_sync(FULL_MASK, vall, offset)  
        offset<<= 1
    end
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
    i = (threadIdx().x* 16) + ((blockIdx().x - 1) *16) * (blockDim().x)# used as a basis to get data we want from global memory
    blockId = blockIdx().x
   wid, lane = fldmod1(threadIdx().x,32)

   #shared memory for  stroing intermidiate data per lane  
   shmem = @cuStaticSharedMem(UInt16, (513,3))
   #for storing results from warp reductions
   shmemSum = @cuStaticSharedMem(UInt16, (33,3))
    #incrementing - taking multiple datapoints per lane  

   @unroll for k in 0:15
    incr_shmem(threadIdx().x,goldBoolGPU[i+k],segmBoolGPU[i+k],shmem)
   end#for 
   sync_warp()
   #reducing across the warp
   @inbounds sumFn = reduce_warp(shmem[threadIdx().x,1],32)
   @inbounds sumFp = reduce_warp(shmem[threadIdx().x,2],32)
   @inbounds sumTp = reduce_warp(shmem[threadIdx().x,3],32)

   if(lane==1)
        if(sumFp>0)
        # 100*500/9018230183092381*62387462384623846/9342903840923840*982734982734
       #CUDA.@cuprint "sumFp $(sumFp) wid $(wid)    "
        end
    @inbounds shmemSum[wid,1]= sumFn
#     end  
#    if(lane==2) 
        @inbounds shmemSum[wid,2]= sumFp
#    end     
#    if(lane==3)
        @inbounds shmemSum[wid,3]= sumTp
    end#if  
sync_threads()
#now all data about of intrest should be in  shared memory so we will get all rsults from warp reduction in the shared memory
    # in case we have only 32 warps as we set we will not go out of bounds
    if(wid==2 )
        vallTp = reduce_warp(shmemSum[lane,3],32 )
        #probably we do not need to sync warp as shfl dow do it for us        

        if(lane==1)
            @inbounds @atomic tp[]+=vallTp
            #@inbounds intermediateResTp[blockId]=vallTp
        end
    end  
    
    if(wid==3 )   
        vallFp = reduce_warp(shmemSum[lane,2],32 )
         

        if(lane==1)
            @inbounds @atomic fp[]+=vallFp
            #@inbounds intermediateResFp[blockId]=vallFp
        end 
    end 

    if(wid==5)  
            vallFn = reduce_warp(shmemSum[lane,1],32 )
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
blockss = Int64(round((length(FlattGoldGPU)/512)/16))-1
#@benchmark CUDA.@sync 
 @cuda threads=512 blocks=blockss getBlockTpFpFn(FlattGoldGPU,FlattSegGPU,tp,tn,fp,fn, intermediateResTp,intermediateResFp,intermediateResFn) 
tp[1]
fp[1]

tp[1] ==tpTotalTrue && fp[1] ==fpTotalTrue && fn[1] ==fnTotalTrue #tn[1] == tnTotalTrue && 

tp[1]
fp[1]
fn[1]

  @device_code_warntype interactive=true @cuda getBlockTpFpFn(FlattGoldGPU,FlattSegGPU,tp,tn,fp,fn, intermediateResTp,intermediateResFp,intermediateResFn)

## using occupancy API to calculate  threads number, block number etc...
#args = (FlattGoldGPU,FlattSegGPU,tp,tn,fp,fn, intermediateResTp,intermediateResFp,intermediateResFn)
calcBlocks,valcThreads,maxBlocks = computeBlocksFromOccupancy(args,3 )




intermediateResTp,intermediateResFp,intermediateResFn
sum(intermediateResTp)
# from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/test/execution.jl
sum(FlattGoldGPU)
kernel = cufunction(kernelFunction, (FlattGoldGPU,FlattSegGPU,tp,tn,fp,fn, intermediateResTp,intermediateResFp,intermediateResFn))


x,y = fldmod1(68,32)




