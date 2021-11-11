######### getting all together in Housedorff 

using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils,Main.ResultListUtils, Main.Housdorff


mainArrDims= (67,177,90);
mainArrCPU= falses(mainArrDims);
refArrCPU = falses(mainArrDims);
##### we will create two planes 20 units apart from each 
mainArrCPU[10:50,10:50,10]= true
refArrCPU[10:50,10:50,30]= true


    
mainArrGPU = CuArray(mainArrCPU);
refArrGPU= CuArray(mainArrCPU);
 





