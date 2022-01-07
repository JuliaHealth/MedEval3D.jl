using CUDA
include("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
include("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
using ..BasicPreds, ..CUDAGpuUtils,Cthulhu,BenchmarkTools , CUDA


goldBoolGPU= CuArray(trues(16,16,16));

segmBoolGPU= CuArray(trues(16,16,16));

fn = CuArray([0])

function kernelFunct(goldBoolGPU::CuDeviceArray{Bool, 3, 1}, segmBoolGPU::CuDeviceArray{Bool, 3, 1},fn)

    mainAfDiv, zz = fldmod1(blockIdx().x,8)

    i= (mainAfDiv-1) * blockDimX() + threadIdxX()

    j = (blockIdx().y-1) * blockDimY() + threadIdxY()

    z = (blockIdx().z-1) * blockDimZ() + threadIdxZ() 



    CUDA.@cuprint """ i $(i)  j $(j) z $(z)   blockIdx().x $(blockIdx().x)  blockDimX()  $(blockDimX())   threadIdxX() $(threadIdxX())     blockIdx().y $(blockIdx().y)  blockDimY()  $(blockDimY())   threadIdxY() $(threadIdxY())  blockIdx().z $(blockIdx().z)  blockDimZ()  $(blockDimZ())   threadIdxZ() $(threadIdxZ()) \n """

    sync_threads()

    # if (goldBoolGPU[i,j,z] & !segmBoolGPU[i,j,z] )

    #    CUDA.@atomic fn[]+=1    

    #     end

    return  

    end

@cuda threads=(2, 2,2) blocks=(2,2,2)  kernelFunct(goldBoolGPU,segmBoolGPU,fn)





using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
using ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils,..ResultListUtils, ..Housdorff

using CUDA
dataBdim= (32,24,32)
fp = CUDA.zeros(UInt16,1)
sumInBits = (dataBdim[1]+2)+(dataBdim[2]+2)+(dataBdim[3]+2)+dataBdim[1]+dataBdim[2]+dataBdim[3]
shmemSum = cld(sumInBits,8)#in bytes
function testKernelA(dataBdim,fp)
    resShmem =  @cuDynamicSharedMem(Bool,((dataBdim[1]+2),(dataBdim[2]+2),(dataBdim[3]+2))) 
       sourceShmem =  @cuDynamicSharedMem(Bool,(dataBdim[1],dataBdim[2],dataBdim[3]))
    #   
    for i in 1:(dataBdim[1]+2),j in 1:(dataBdim[2]+2), n in 1:(dataBdim[3]+2)
        resShmem[i,j,n]=false
    end
 
    for i in 1:(dataBdim[1]),j in 1:(dataBdim[2]), n in 1:(dataBdim[3])
        sourceShmem[i,j,n]=false
    end
    fp[1]=1
return
end
@cuda threads=(32,5) blocks=(2) shmem=shmemSum  testKernelA(dataBdim,fp)
fp[1]





using CUDA
dataBdim= (32,24,32)
fp = CUDA.zeros(UInt16,1)
sumInBits = (dataBdim[1]+2)*(dataBdim[2]+2)*(dataBdim[3]+2)+dataBdim[1]*dataBdim[2]*dataBdim[3]
shmemSum = cld(sumInBits,8)#in bytes
function testKernelA(dataBdim,fp)
    resShmem =  @cuStaticSharedMem(Bool,(34,26,34)) 
    #sourceShmem =  @cuStaticSharedMem(Bool,(32,24,32))
    #   
    for i in 1:(dataBdim[1]+2),j in 1:(dataBdim[2]+2), n in 1:(dataBdim[3]+2)
        resShmem[i,j,n]=false
    end
 
    # for i in 1:(dataBdim[1]),j in 1:(dataBdim[2]), n in 1:(dataBdim[3])
    #     sourceShmem[i,j,n]=false
    # end
    fp[1]=1
return
end
@cuda threads=(32,5) blocks=(2) shmem=shmemSum  testKernelA(dataBdim,fp)
fp[1]