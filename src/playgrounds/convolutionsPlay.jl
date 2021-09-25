using CUDA
include("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
include("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
using Main.BasicPreds, Main.CUDAGpuUtils,Cthulhu,BenchmarkTools , CUDA


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

    #     @atomic fn[]+=1    

    #     end

    return  

    end

@cuda threads=(2, 2,2) blocks=(2,2,2)  kernelFunct(goldBoolGPU,segmBoolGPU,fn)

fn