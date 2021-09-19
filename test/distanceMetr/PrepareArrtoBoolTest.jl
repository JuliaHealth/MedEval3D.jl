using Test,Revise
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\gpuUtils.jl")


includet("C:/GitHub/GitHub/NuclearMedEval/src/kernels/PrepareArrtoBool.jl")
using Main.PrepareArrtoBool, Main.GPUutils
using CUDA

@testset "allocateMomory" begin 
    sliceW = 512
    sliceH = 512
    arrGold = CUDA.zeros(sliceW,sliceH,800);
    loopNumb, indexCorr= getKernelContants(1024, sliceW*sliceH)
end