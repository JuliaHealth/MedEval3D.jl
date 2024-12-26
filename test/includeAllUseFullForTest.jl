
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



includet("../src/utils/CUDAAtomicUtils.jl")

includet("../src/utils/BitWiseUtils.jl")
includet("../src/structs/BasicStructs.jl")
includet("../src/utils/CUDAGpuUtils.jl")
includet("../src/utils/IterationUtils.jl")
includet("../src/utils/ReductionUtils.jl")
includet("../src/utils/MemoryUtils.jl")
includet("../src/distanceMetrics/Housdorff/verB/MetaDataUtils.jl")
includet("../src/distanceMetrics/Housdorff/verB/ResultListUtils.jl")

includet("../src/overLap/MainOverlap.jl")
includet("../src/PairCounting/RandIndex.jl")

includet("../src/Probabilistic/ProbabilisticMetrics.jl")
includet("../src/InformationTheorhetic/InformationTheorhetic.jl")
includet("../src/volume/VolumeMetric.jl")

includet("../src/distanceMetrics/MeansMahalinobis.jl")
includet("../src/distanceMetrics/Housdorff/verB/HFUtils.jl")

includet("../src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")
includet("../src/distanceMetrics/Housdorff/verB/WorkQueueUtils.jl")
includet("../src/distanceMetrics/Housdorff/verB/ProcessMainDataVerB.jl")
includet("../test/GPUtestUtils.jl")
includet("../src/distanceMetrics/Housdorff/verB/ScanForDuplicates.jl")
includet("../src/distanceMetrics/Housdorff/verB/MetadataAnalyzePass.jl")
includet("../src/distanceMetrics/Housdorff/verB/MainLoopKernel.jl")
includet("../src/distanceMetrics/Housdorff/verB/Housdorff.jl")

includet("../src/kernels/TpfpfnKernel.jl")
includet("../src/kernels/InterClassCorrKernel.jl")
includet("../src/MainAbstractions.jl")