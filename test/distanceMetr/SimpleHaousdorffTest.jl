using Test,Revise
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\gpuUtils.jl")

includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/SimplerHousdorff.jl")
using Main.SimplerHousdorff, Main.GPUutils
using CUDA

# @testset "getBlockNumb" begin 
# arrGold = CUDA.zeros(1023,200,800);
#     @test  Main.SimplerHousdorff.getBlockNumb(size(arrGold))==(800, 1023  )
#     CUDA.reclaim()# just to destroy from gpu our dummy data
#     arrGold = CUDA.zeros(1028,200,2000);
#     @test  Main.SimplerHousdorff.getBlockNumb(size(arrGold))==(4000,1024)
#     CUDA.reclaim()# just to destroy from gpu our dummy data
# end # 
    

@testset "allocateMomory" begin 
    arrGold = CUDA.zeros(1024,200,800);
    @test  size(Main.SimplerHousdorff.allocateMomory(size(arrGold))) == (32, cld(200,32), cld(800,32),6)
    CUDA.reclaim()# just to destroy from gpu our dummy data

end # 

