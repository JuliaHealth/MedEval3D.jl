
using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAAtomicUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MetaDataUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\MetadataAnalyzePass.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils
using Main.MetadataAnalyzePass


#########metaDataWarpIter
singleVal = CUDA.zeros(14)

threads=(32,5)
blocks =8
mainArrDims= (516,523,826)
datBdim = (43,21,17)
 metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,datBdim),1:17,2:18,4:10,: );
#metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,datBdim),1:9,2:3,4:6,: );
metaDataDims=size(metaData)
loopXMeta= fld(metaDataDims[1],threads[1])
loopYZMeta= fld(metaDataDims[2]*metaDataDims[3],blocks )
metaDataDims,loopXMeta,loopYZMeta

function metaDataWarpIterKernel(singleVal,metaDataDims,loopXMeta,loopYZMeta)

  
    MetadataAnalyzePass.@metaDataWarpIter( metaDataDims,loopXMeta,loopYZMeta,
    begin
      @ifY 1 @atomic singleVal[]+=1
      #@ifX 1 CUDA.@cuprint "   xMeta $(xMeta)  yMeta $(yMeta)  zMeta $(zMeta) id  idX $(blockIdxX()) \n"   

    # @ifXY 1 1    CUDA.@cuprint "linIndex $(linIndex)"
    #CUDA.@cuprint "linIndex $(linIndex) \n "
end)
    
    return
end
@cuda threads=threads blocks=blocks metaDataWarpIterKernel(singleVal,metaDataDims,loopXMeta,loopYZMeta)
@test singleVal[1]==metaDataDims[1]*metaDataDims[2]*metaDataDims[3]











####### exOnWarpIfNotFull





using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAAtomicUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MetaDataUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\MetadataAnalyzePass.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils
using Main.MetadataAnalyzePass

threads=(32,7)
blocks=(1)
resCounter= CUDA.zeros(Int32,1)
numbOfLoops = 200
function exOnWarpKernelA(resCounter,numbOfLoops)
  for i in 1:numbOfLoops
      MetadataAnalyzePass.@exOnWarp i @ifX 1  @atomic resCounter[1]+=1
      # MetadataAnalyzePass.@exOnWarp i @ifX 1  CUDA.@cuprint "i $(i)  mod $(mod(i,blockDimY() ))  idY   $(threadIdxY()) \n"   #@atomic resCounter[1]+=1
      
  end
  return
end 
    @cuda threads=threads blocks=blocks exOnWarpKernelA(resCounter,numbOfLoops)

    @test resCounter[1]==numbOfLoops





using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAAtomicUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MetaDataUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\MetadataAnalyzePass.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils
using Main.MetadataAnalyzePass

threads=(17,13)
blocks=(1)
resCounter= CUDA.zeros(Int64,1)
function exOnWarpIfBoolKernelA(resCounter)
    #bool true
  for i in 1:200
      MetadataAnalyzePass.@exOnWarpIfBool true i @ifX 1 @atomic resCounter[1]+=1
  end
  #bool false
    for i in 1:200
      MetadataAnalyzePass.@exOnWarpIfBool false i @ifX 1 @atomic resCounter[1]+=1
  end
  return
end 
    @cuda threads=threads blocks=blocks exOnWarpIfBoolKernelA(resCounter)

    @test resCounter[1]==200



##################### load counters


using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAAtomicUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MetaDataUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\MetadataAnalyzePass.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils
using Main.MetadataAnalyzePass


#########metaDataWarpIter
singleVal = CUDA.zeros(14)

threads=(32,5)
blocks =8
mainArrDims= (516,523,826)
datBdim = (43,21,17)
 metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,datBdim),1:7,2:11,4:13,: );
#metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,datBdim),1:9,2:3,4:6,: );
metaDataDims=size(metaData)
maxLinIndex = metaDataDims[1]*metaDataDims[2]*metaDataDims[3]
metaDataIterLoops = cld( maxLinIndex,blocks*32)
function metaDataWarpIterKernel(singleVal,metaDataIterLoops, maxLinIndex)

  
    MetadataAnalyzePass.@metaDataWarpIter( metaDataIterLoops, maxLinIndex,
    begin
      @ifY 1 @atomic singleVal[]+=1
    # @ifXY 1 1    CUDA.@cuprint "linIndex $(linIndex)"
    #CUDA.@cuprint "linIndex $(linIndex) \n "
end)
    
    return
end
@cuda threads=threads blocks=blocks metaDataWarpIterKernel(singleVal,metaDataIterLoops, maxLinIndex)
@test singleVal[1]==metaDataDims[1]*metaDataDims[2]*metaDataDims[3]


