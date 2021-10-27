using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.ResultListUtils, Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils, Main.ScanForDuplicates

resList = allocateResultList(100,100);

resList[1,:]= [1,2,3]

carts = CartesianIndices(zeros(200,200,200))

mappedA = map(ind-> getResLinIndex([ind[1],ind[2],ind[3],1,1,1], (200,200,200)) ,carts)
mappedB = map(ind-> getResLinIndex([ind[1],ind[2],ind[3],0,1,1], (200,200,200)) ,carts)

@test length(unique(mappedA))== 200*200*200
@test length(unique(vec(vcat(mappedA,mappedB ))))== 200*200*200*2
getResLinIndex()