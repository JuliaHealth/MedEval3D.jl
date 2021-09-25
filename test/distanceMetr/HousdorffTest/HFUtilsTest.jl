#module HFUtilsTest


using  Test, Revise 
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\mainHouseDorffKernel\\HFUtils.jl")

using Main.HFUtils
using Main.CUDAGpuUtils,Cthulhu,BenchmarkTools , CUDA


@testset "clearMainShmem" begin 

    testArr = CUDA.zeros(Bool,34,34,34);
    function testKernForClearShmem(testArrInn)
        resShmem =  @cuStaticSharedMem(Bool,(34,34,34))#+2 in order to get the one padding 
        resShmem[threadIdxX(),threadIdxY(),1]= 1
        testArrInn[threadIdxX(),threadIdxY(),1]= 1
        HFUtils.clearMainShmem( resShmem, threadIdxX(),threadIdxY())
        HFUtils.clearMainShmem( testArrInn, threadIdxX(),threadIdxY())
               
        return
    end    
    @cuda threads=(32,32) blocks=1 testKernForClearShmem(testArr) 
    testArr[:,:,1]
    sum(testArr)

    CUDA.reclaim()# just to destroy from gpu our dummy data







end # 
