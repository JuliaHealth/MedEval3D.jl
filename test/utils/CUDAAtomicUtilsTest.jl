using Revise, Parameters, Logging, Test
using CUDA
# includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
# includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
# includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
# includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
# includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
# includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
includet("../../src/utils/CUDAAtomicUtils.jl")

#using ..BasicPreds, ..CUDAGpuUtils , ..IterationUtils,..ReductionUtils , ..MemoryUtils, ..CUDAAtomicUtils
using ..CUDAAtomicUtils

result = CUDA.zeros(1)
function testAddAtomicKernel(result)
    old = atomicallyAddOne(result)
    CUDA.@cuprint " old $(old)"
    return
end
@cuda threads=32 blocks=1 testAddAtomicKernel(result)
@test result[1]==32



resultVV = CUDA.zeros(1)
function testAddAtomicKernelVV(result)
    atomicAdd(result,2)
    return
end
@cuda threads=32 blocks=1 testAddAtomicKernelVV(resultVV)
@test resultVV[1]==64


resultB = CUDA.zeros(3)
function testAddAtomicKernelB(resultB)
    old = atomicallyAddToSpot(resultB,2,1)
    CUDA.@cuprint " old $(old)"

    # CUDA.atomic_add!(pointer(resultB, 2), Float32(1))
    
    return
end
@cuda threads=32 blocks=1 testAddAtomicKernelB(resultB)
@test resultB[2]==32





resultDD = CUDA.zeros(1)
function atomicMinSetKernel(resultDD)
    atomicMinSet(resultDD,5)
    return
end
@cuda threads=32 blocks=1 testAddAtomicKernelB(resultDD)
@test resultDD[1]==0



resultEE = CUDA.ones(3)
function testAddAtomicKernelZ(resultEE)
    #atomicMinSet(Float32,resultEE,2,1)
    #CUDA.atomic_min!(pointer(resultEE, 2), Float32(0))
    atomicMinSet(resultEE,0)
    return
end
@cuda threads=32 blocks=1 testAddAtomicKernelZ(resultEE)
@test resultEE[1]==0


resultEF = CUDA.ones(3)
function testAddAtomicKernelZR(resultEF)
    #atomicMinSet(Float32,resultEE,2,1)
    #CUDA.atomic_min!(pointer(resultEE, 2), Float32(0))
    atomicMinSet(resultEF,0,2)
    return
end
@cuda threads=32 blocks=1 testAddAtomicKernelZR(resultEF)
@test resultEF[2]==0



resultFo = CUDA.zeros(2)
function atomicMaxSetKernel(resultFo)
    atomicMaxSet(resultFo,10)
    return
end
@cuda threads=32 blocks=1 atomicMaxSetKernel(resultFo)
@test resultFo[1]==10



resultFo2 = CUDA.zeros(2)
function atomicMaxSetKernel(resultFo2)
    atomicMaxSet(resultFo2,10,2)
    return
end
@cuda threads=32 blocks=1 atomicMaxSetKernel(resultFo2)
@test resultFo2[2]==10
