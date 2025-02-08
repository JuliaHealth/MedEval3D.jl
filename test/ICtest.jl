using Revise, Parameters, Logging, Test
using CUDA
includet("./includeAllUseFullForTest.jl")
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils, ..BasicStructs
using Shuffle,..ResultListUtils, ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils, ..ScanForDuplicates
using ..MainOverlap, ..RandIndex , ..ProbabilisticMetrics , ..VolumeMetric ,..InformationTheorhetic
using ..CUDAAtomicUtils, ..TpfpfnKernel, ..InterClassCorrKernel

dimx = 529
dimy = 556
dimz = 339
arrGold = CUDA.fill(UInt8(1), (dimx, dimy, dimz))
arrAlgo = CUDA.fill(UInt8(1), (dimx, dimy, dimz))
numberToLooFor = UInt8(1)

# Calculate ICC first to ensure kernel execution
globalICC = InterClassCorrKernel.calculateInterclassCorr(arrGold, arrAlgo, numberToLooFor)
CUDA.synchronize()

# Now safely retrieve the sums using proper scalar access
sum_of_gold, sum_of_segm, _, _, _ = InterClassCorrKernel.prepareInterClassCorrKernel(arrGold, arrAlgo, numberToLooFor)

gold_sum = CUDA.@allowscalar sum_of_gold[]
segm_sum = CUDA.@allowscalar sum_of_segm[]

println("Gold sum: ", gold_sum)
println("Segm sum: ", segm_sum)
println("Global ICC: ", globalICC)