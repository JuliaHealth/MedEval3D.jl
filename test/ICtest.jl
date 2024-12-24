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
arrGold = CUDA.ones(UInt8, (dimx,dimy,dimz) )
arrAlgo = CUDA.ones(UInt8, (dimx,dimy,dimz) )
numberToLooFor = UInt8(1)
argsMain, threadsMain,  blocksMain,threadsMean,blocksMean,argsMean, totalNumbOfVoxels=InterClassCorrKernel.prepareInterClassCorrKernel(arrGold ,arrAlgo,numberToLooFor)

globalICC=calculateInterclassCorr(arrGold,arrAlgo,argsMain, threadsMain,  blocksMain,threadsMean,blocksMean,argsMean, totalNumbOfVoxels)::Float64


Int64(argsMain[1][1]-dimx*dimy*dimz)
argsMain[2][1]==dimx*dimy*dimz
