
    using Revise, Parameters, Logging, Test
    using CUDA
    includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
    using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
    using Main.BitWiseUtils,Main.ResultListUtils, Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils, Main.ScanForDuplicates

numb = 0
pos = 2
@test !isBit1AtPos(numb,pos)

@setBitTo1(numb,pos)

@test isBit1AtPos(numb,pos)
@test !isBit1AtPos(numb,pos-1)
@test !isBit1AtPos(numb,pos+1)

numb = 0
pos = 2

@setBitTo(numb,pos,true)
@test isBit1AtPos(numb,pos)

@setBitTo(numb,pos,false)
@test !isBit1AtPos(numb,pos)

numbs=[2]
@setBitTo(numbs[1],pos,true)


##### bitDilatate
numb = 0
@setBitTo(numb,1,true)
@setBitTo(numb,5,true)

@bitDilatate(numb)

@test isBit1AtPos(numb,1)
@test isBit1AtPos(numb,2)
@test !isBit1AtPos(numb,3)
@test isBit1AtPos(numb,4)
@test isBit1AtPos(numb,5)
@test isBit1AtPos(numb,6)
@test !isBit1AtPos(numb,7)



