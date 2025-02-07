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
args, workgroup_size, num_workgroups, total_num_voxels=InterClassCorrKernel.prepareInterClassCorrKernel(arrGold ,arrAlgo,numberToLooFor)
globalICC = InterClassCorrKernel.calculateInterclassCorr(arrGold, arrAlgo, numberToLooFor)

Int64(args[1][1] - dimx * dimy * dimz)
args[1][2] == dimx * dimy * dimz