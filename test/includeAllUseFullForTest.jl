
using Revise, Parameters, Logging, Test
using CUDA
# include("/home/jakub/Documents/NuclearMedEval/src/utils/CUDAAtomicUtils.jl")


# includet("/home/jakub/Documents/NuclearMedEval/src/utils/BitWiseUtils.jl")
# includet("/home/jakub/Documents/NuclearMedEval/src/structs/BasicStructs.jl")
# includet("/home/jakub/Documents/NuclearMedEval/src/utils/CUDAGpuUtils.jl")
# includet("/home/jakub/Documents/NuclearMedEval/src/utils/IterationUtils.jl")
# includet("/home/jakub/Documents/NuclearMedEval/src/utils/ReductionUtils.jl")
# includet("/home/jakub/Documents/NuclearMedEval/src/utils/MemoryUtils.jl")
# includet("/home/jakub/Documents/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MetaDataUtils.jl")
# includet("/home/jakub/Documents/NuclearMedEval/src/distanceMetrics/Housdorff/verB/ResultListUtils.jl")

# includet("/home/jakub/Documents/NuclearMedEval/src/overLap/MainOverlap.jl")
# includet("/home/jakub/Documents/NuclearMedEval/src/PairCounting/RandIndex.jl")

# includet("/home/jakub/Documents/NuclearMedEval/src/Probabilistic/ProbabilisticMetrics.jl")
# includet("/home/jakub/Documents/NuclearMedEval/src/InformationTheorhetic/InformationTheorhetic.jl")
# includet("/home/jakub/Documents/NuclearMedEval/src/volume/VolumeMetric.jl")

# includet("/home/jakub/Documents/NuclearMedEval/src/distanceMetrics/MeansMahalinobis.jl")
# includet("/home/jakub/Documents/NuclearMedEval/src/distanceMetrics/Housdorff/verB/HFUtils.jl")

# includet("/home/jakub/Documents/NuclearMedEval/src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")
# includet("/home/jakub/Documents/NuclearMedEval/src/distanceMetrics/Housdorff/verB/WorkQueueUtils.jl")
# includet("/home/jakub/Documents/NuclearMedEval/src/distanceMetrics/Housdorff/verB/ProcessMainDataVerB.jl")
# includet("/home/jakub/Documents/NuclearMedEval/test/GPUtestUtils.jl")
# includet("/home/jakub/Documents/NuclearMedEval/src/distanceMetrics/Housdorff/verB/ScanForDuplicates.jl")
# includet("/home/jakub/Documents/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MetadataAnalyzePass.jl")
# includet("/home/jakub/Documents/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MainLoopKernel.jl")
# includet("/home/jakub/Documents/NuclearMedEval/src/distanceMetrics/Housdorff/verB/Housdorff.jl")

# includet("/home/jakub/Documents/NuclearMedEval/src/kernels/TpfpfnKernel.jl")
# includet("/home/jakub/Documents/NuclearMedEval/src/kernels/InterClassCorrKernel.jl")



includet("C:/GitHub/GitHub/NuclearMedEval/src/utils/CUDAAtomicUtils.jl")

includet("C:/GitHub/GitHub/NuclearMedEval/src/utils/BitWiseUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/structs/BasicStructs.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/utils/CUDAGpuUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/utils/IterationUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/utils/ReductionUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/utils/MemoryUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MetaDataUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/ResultListUtils.jl")

includet("C:/GitHub/GitHub/NuclearMedEval/src/overLap/MainOverlap.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/PairCounting/RandIndex.jl")

includet("C:/GitHub/GitHub/NuclearMedEval/src/Probabilistic/ProbabilisticMetrics.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/InformationTheorhetic/InformationTheorhetic.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/volume/VolumeMetric.jl")

includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/MeansMahalinobis.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/HFUtils.jl")

includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/WorkQueueUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/ProcessMainDataVerB.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/test/GPUtestUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/ScanForDuplicates.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MetadataAnalyzePass.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MainLoopKernel.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/Housdorff.jl")

includet("C:/GitHub/GitHub/NuclearMedEval/src/kernels/TpfpfnKernel.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/kernels/InterClassCorrKernel.jl")
