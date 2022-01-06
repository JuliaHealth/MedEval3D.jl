    using Revise, Parameters, Logging, Test
    using CUDA
    includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
    using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
    using ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils,..ResultListUtils, ..Housdorff
    using ..ResultProcess

fpdummy = 100
fndummy = 100

resList,resListIndicies = allocateResultLists(fpdummy,fndummy)
reference= zeros(UInt8,100)
for i in 1:100
  resList[i,6]=i
  reference[i]=i
end
#only non 0 should be taken into acount

for i in 1:90
  resListIndicies[i]=i
end

function testGetAverage()
  getAverage(resList,resListIndicies,entriesPerBlock,totalLength,iterLoopResList ,globalSum )
return
end

@cuda threads=(16,16) blocks=1 testKernelForPaddingAnalysis(testArrIn,referenceArray,blockBeginingX,blockBeginingY,blockBeginingZ,  resArray,
