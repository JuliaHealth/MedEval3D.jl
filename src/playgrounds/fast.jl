

using Revise, Parameters, Logging
using CUDA
include("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
include("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\gpuUtils.jl")
using Main.BasicPreds, Main.GPUutils,Cthulhu,BenchmarkTools , CUDA






@inline function reduce_warp( vall, lanesNumb)

    offset = UInt32(1)
    while(offset <lanesNumb) 
        vall+=shfl_down_sync(FULL_MASK, vall, offset)  
        offset<<= 1
    end
    return vall

end


@inline function incr_shmem( primi::Int64,boolGold::Bool,boolSegm::Bool,shmem )
   @inbounds shmem[ primi, (boolGold & boolSegm + boolSegm +1) ]+=(boolGold | boolSegm)
   return true
end



function getBlockTpFpFn(goldBoolGPU::CuDeviceArray{Bool,1, 1}

    , segmBoolGPU::CuDeviceArray{Bool, 1, 1}

    ,tp,tn,fp,fn)

i = (threadIdx().x* 16) + ((blockIdx().x - 1) *16) * (blockDim().x)# used as a basis to get data we want from global memory
wid, lane = fldmod1(threadIdx().x,32)
shmem = @cuStaticSharedMem(UInt16, (513,3))

incr_shmem(threadIdx().x,goldBoolGPU[i],segmBoolGPU[i],shmem)
incr_shmem(threadIdx().x,goldBoolGPU[i+1],segmBoolGPU[i+1],shmem)

sync_warp()
@inbounds sumTp = reduce_warp(shmem[threadIdx().x,3],32)
if(lane==1)
    if(sumTp>0)
    # !!!!!!!!!! HERE is this magical print statement!!!!!!!!!!!!
   CUDA.@cuprint "sumFp $(sumTp) wid $(wid)    "
    end
    @inbounds @atomic tp[]+=sumTp
end#if  
return  
end

nx=32

    ny=32

    nz=32

    #first we initialize the metrics on CPU so we will modify them easier

    goldBool= falses(nx,ny,nz); #mimicks gold standard mask

    segmBool= falses(nx,ny,nz); #mimicks mask     

# so we  have 2 cubes that are overlapped in their two thirds

    cartTrueGold =  CartesianIndices(zeros(3,3,5) ).+CartesianIndex(5,5,5);

    cartTrueSegm =  CartesianIndices(zeros(3,3,3) ).+CartesianIndex(4,5,5); 

    goldBool[cartTrueGold].=true

    segmBool[cartTrueSegm].=true


    #for storing output total first tp than TN than Fp and Fn
    tp= CuArray([0]);
    tn= CuArray([0]);
    fp= CuArray([0]);
    fn = CuArray([0]);


FlattG = push!(vec(goldBool),false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false);

FlattSeg = push!(vec(segmBool),false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false);

FlattGoldGPU= CuArray( FlattG)

FlattSegGPU= CuArray( FlattSeg )

goldBoolGPU= CuArray( goldBool)

segmBoolGPU= CuArray( segmBool )

blockss = Int64(round((length(FlattGoldGPU)/512)/16))-1
#@benchmark CUDA.@sync 
 @cuda threads=512 blocks=blockss getBlockTpFpFn(FlattGoldGPU,FlattSegGPU,tp,tn,fp,fn)
 tp[1]

 tp[1] ==tpTotalTrue && fp[1] ==fpTotalTrue && fn[1] ==fnTotalTrue 




 