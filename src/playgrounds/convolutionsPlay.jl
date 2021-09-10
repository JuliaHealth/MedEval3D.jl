using CUDA
include("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
include("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\gpuUtils.jl")
using Main.BasicPreds, Main.GPUutils,Cthulhu,BenchmarkTools , CUDA


goldBoolGPU= CuArray(trues(16,16,16));

segmBoolGPU= CuArray(trues(16,16,16));

fn = CuArray([0])

function kernelFunct(goldBoolGPU::CuDeviceArray{Bool, 3, 1}, segmBoolGPU::CuDeviceArray{Bool, 3, 1},fn)

    mainAfDiv, zz = fldmod1(blockIdx().x,8)

    i= (mainAfDiv-1) * blockDim().x + threadIdx().x

    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    z = (blockIdx().z-1) * blockDim().z + threadIdx().z 



    CUDA.@cuprint """ i $(i)  j $(j) z $(z)   blockIdx().x $(blockIdx().x)  blockDim().x  $(blockDim().x)   threadIdx().x $(threadIdx().x)     blockIdx().y $(blockIdx().y)  blockDim().y  $(blockDim().y)   threadIdx().y $(threadIdx().y)  blockIdx().z $(blockIdx().z)  blockDim().z  $(blockDim().z)   threadIdx().z $(threadIdx().z) \n """

    sync_threads()

    # if (goldBoolGPU[i,j,z] & !segmBoolGPU[i,j,z] )

    #     @atomic fn[]+=1    

    #     end

    return  

    end

@cuda threads=(2, 2,2) blocks=(2,2,2)  kernelFunct(goldBoolGPU,segmBoolGPU,fn)

fn