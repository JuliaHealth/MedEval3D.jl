using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAAtomicUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MetaDataUtils.jl")
using Main.PrepareArrtoBool, Main.CUDAGpuUtils, Main.PrepareArrtoBool, Main.CUDAAtomicUtils, Main.MetaDataUtils
using CUDA


localQuesValues = CUDA.zeros(Float32,14)
threads=(32,5)
blocks =3

mainArrDims= (67,78,90)
dataBdim = (17,7,12)

metaDataDims= (cld(mainArrDims[1],dataBdim[1] ),cld(mainArrDims[2],dataBdim[2]),cld(mainArrDims[3],dataBdim[3]))
metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);
size(metaData)









