using Test,Revise
includet("./src/utils/CUDAGpuUtils.jl")

includet("./src/distanceMetrics/SimplerHousdorff.jl")
using ..SimplerHousdorff, ..CUDAGpuUtils
using CUDA

# @testset "getBlockNumb" begin 
# arrGold = CUDA.zeros(1023,200,800);
#     @test  ..SimplerHousdorff.getBlockNumb(size(arrGold))==(800, 1023  )
#     CUDA.reclaim()# just to destroy from gpu our dummy data
#     arrGold = CUDA.zeros(1028,200,2000);
#     @test  ..SimplerHousdorff.getBlockNumb(size(arrGold))==(4000,1024)
#     CUDA.reclaim()# just to destroy from gpu our dummy data
# end # 
    

@testset "allocateMomory" begin 
    arrGold = CUDA.zeros(1024,200,800);
    @test  size(SimplerHousdorff.allocateMemory(size(arrGold))) == (32, cld(200,32), cld(800,32),6)
    CUDA.reclaim()# just to destroy from gpu our dummy data

end # 

