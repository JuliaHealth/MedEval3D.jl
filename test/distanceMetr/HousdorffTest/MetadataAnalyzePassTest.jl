
using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\MetadataAnalyzePass.jl")
using Main.BasicPreds, Main.CUDAGpuUtils , Main.MeansMahalinobis, Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils
using Main.MetadataAnalyzePass


#########metaDataWarpIter

threads=(17,13)
blocks=(27)

function testmetaDataWarpIterKernelA(metaData)
    
MetadataAnalyzePass.metaDataWarpIter(metaData,
  ,begin
end)
        return
    end

    @cuda threads=threads blocks=blocks testmetaDataWarpIterKernelA(args...)

    @test 1==1


####### exOnWarpIfNotFull
threads=(17,13)
blocks=(27)
resCounter= CUDA.zeros(UInt32,1)
function exOnWarpIfBoolKernelA(resCounter)
    #bool true
  for i in 1:200
      MetadataAnalyzePass.@exOnWarpIfBool true i @ifX 1 @atomic resCounter[1]+=1
  end
  #bool false
    for i in 1:200
      MetadataAnalyzePass.@exOnWarpIfBool false i @ifX 1 @atomic resCounter[1]+=1
  end
  
    @cuda threads=threads blocks=blocks exOnWarpIfBoolKernelA(args...)

    @test resCounter[1]==200



