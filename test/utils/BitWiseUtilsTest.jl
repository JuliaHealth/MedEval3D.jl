
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

numbB= @bitDilatate(numb)

@test isBit1AtPos(numbB,1)
@test isBit1AtPos(numbB,2)
@test !isBit1AtPos(numbB,3)
@test isBit1AtPos(numbB,4)
@test isBit1AtPos(numbB,5)
@test isBit1AtPos(numbB,6)
@test !isBit1AtPos(numbB,7)

@test !isBit1AtPos(numb,2)


#### bitPassOnes


source = 0
target = 0
@setBitTo(source,1,true)
@setBitTo(source,5,true)
@setBitTo(target,4,true)
new = @bitPassOnes(source,target)

@test isBit1AtPos(new,1)
@test !isBit1AtPos(new,2)
@test !isBit1AtPos(new,3)
@test isBit1AtPos(new,4)
@test isBit1AtPos(new,5)


# using Revise, Parameters, Logging, Test
# using CUDA
# includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")

# using CUDA, Main.CUDAGpuUtils
# function simpleShuffle()
#     xx = threadIdxX()
#     xx+=shfl_down_sync(FULL_MASK, xx, 1) 
#     CUDA.@cuprint "xx $(xx) \n"

#     return
# end

# @cuda threads=(32,1) blocks=1 simpleShuffle()