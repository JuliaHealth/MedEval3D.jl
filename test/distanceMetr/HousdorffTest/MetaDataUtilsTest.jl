using Revise, Parameters, Logging, Test
using CUDA
includet("./src/utils/CUDAAtomicUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("./src/structs/BasicStructs.jl")
includet("./src/utils/CUDAGpuUtils.jl")
includet("./src/utils/IterationUtils.jl")
includet("./src/utils/ReductionUtils.jl")
includet("./src/utils/MemoryUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")
includet("./src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")
includet("./src/distanceMetrics/Housdorff/verB/MetaDataUtils.jl")
using ..PrepareArrtoBool, ..CUDAGpuUtils, ..PrepareArrtoBool, ..CUDAAtomicUtils, ..MetaDataUtils
using CUDA


localQuesValues = CUDA.zeros(Float32,14)
threads=(32,5)
blocks =3

mainArrDims= (67,78,90)
dataBdim = (17,7,12)

metaDataDims= (cld(mainArrDims[1],dataBdim[1] ),cld(mainArrDims[2],dataBdim[2]),cld(mainArrDims[3],dataBdim[3]))
metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);
size(metaData)