using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils, Main.BasicStructs
using Shuffle,Main.ResultListUtils, Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils, Main.ScanForDuplicates
using Main.MainOverlap, Main.RandIndex , Main.ProbabilisticMetrics , Main.VolumeMetric ,Main.InformationTheorhetic
using Main.CUDAAtomicUtils, Main.TpfpfnKernel, Main.InterClassCorrKernel


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
