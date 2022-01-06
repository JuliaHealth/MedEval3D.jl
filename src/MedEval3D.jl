
module MedEval3D

include(joinpath("utils","CUDAAtomicUtils.jl"))

include(joinpath( "utils","BitWiseUtils.jl"))
include(joinpath( "structs","BasicStructs.jl"))
include(joinpath( "utils","CUDAGpuUtils.jl"))
include(joinpath( "utils","IterationUtils.jl"))
include(joinpath( "utils","ReductionUtils.jl"))
include(joinpath( "utils","MemoryUtils.jl"))
include(joinpath( "distanceMetrics","Housdorff","verB","MetaDataUtils.jl"))
include(joinpath( "distanceMetrics","Housdorff","verB","ResultListUtils.jl"))

include(joinpath( "overLap","MainOverlap.jl"))
include(joinpath( "PairCounting","RandIndex.jl"))

include(joinpath( "Probabilistic","ProbabilisticMetrics.jl"))
include(joinpath( "InformationTheorhetic","InformationTheorhetic.jl"))
include(joinpath( "volume","VolumeMetric.jl"))

include(joinpath( "distanceMetrics","MeansMahalinobis.jl"))
include(joinpath( "distanceMetrics","Housdorff","verB","HFUtils.jl"))

include(joinpath( "distanceMetrics","Housdorff","verB","PrepareArrtoBool.jl"))
include(joinpath( "distanceMetrics","Housdorff","verB","WorkQueueUtils.jl"))
include(joinpath( "distanceMetrics","Housdorff","verB","ProcessMainDataVerB.jl"))
include(joinpath( "distanceMetrics","Housdorff","verB","ScanForDuplicates.jl"))
include(joinpath( "distanceMetrics","Housdorff","verB","MetadataAnalyzePass.jl"))
include(joinpath( "distanceMetrics","Housdorff","verB","MainLoopKernel.jl"))
include(joinpath( "distanceMetrics","Housdorff","verB","Housdorff.jl"))

include(joinpath( "kernels","TpfpfnKernel.jl"))
include(joinpath( "kernels","InterClassCorrKernel.jl"))
include(joinpath( "MainAbstractions.jl"))

end#MedEval3D