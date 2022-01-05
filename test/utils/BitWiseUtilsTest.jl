
    using Revise, Parameters, Logging, Test
    using CUDA
    includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
    using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
    using Main.BitWiseUtils,Main.ResultListUtils, Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils, Main.ScanForDuplicates

numb = UInt32(0)
pos = UInt32(2)
@test !isBit1AtPos(numb,pos)

@setBitTo1(numb,pos)

@test isBit1AtPos(numb,pos)
@test !isBit1AtPos(numb,pos-1)
@test !isBit1AtPos(numb,pos+1)

numb = UInt32(0)
pos = UInt32(2)

@setBitTo(numb,pos,true)
@test isBit1AtPos(numb,pos)

@setBitTo(numb,pos,false)
@test !isBit1AtPos(numb,pos)

numbs=[2]
@setBitTo(numbs[1],pos,true)


##### bitDilatate
numb = UInt32(0)
locArr = UInt32(0)
@setBitTo1UINt32(numb,1)
@setBitTo1UINt32(numb,5)
typeof(numb)

numbB= bitDilatate(numb,locArr)

@test isBit1AtPos(numbB,1)
@test isBit1AtPos(numbB,2)
@test !isBit1AtPos(numbB,3)
@test isBit1AtPos(numbB,4)
@test isBit1AtPos(numbB,5)
@test isBit1AtPos(numbB,6)
@test !isBit1AtPos(numbB,7)

@test !isBit1AtPos(numb,2)


#### bitPassOnes


source = UInt32(0)
target = UInt32(0)
@setBitTo1UINt32(source,1)
@setBitTo1UINt32(source,5)
@setBitTo1UINt32(target,4)
new =llvmPassOnes(source,target)

@test isBit1AtPos(new,1)
@test !isBit1AtPos(new,2)
@test !isBit1AtPos(new,3)
@test isBit1AtPos(new,4)
@test isBit1AtPos(new,5)


nnn = UInt8(1)
isBit1AtPos(nnn,1)



function llvmPassOnesBB(source::UInt32,target::UInt32,array::Vector{UInt32})::UInt32
    Base.llvmcall("""
    %3 = or i32 %0, %1
    %2[1]= %3
    ret i32 %3""", UInt32, Tuple{UInt32,UInt32}, source, target)
end


arr = [UInt32(1), UInt32(2)]
source = UInt32(0)
target = UInt32(0)
@setBitTo1UINt32(source,1)
@setBitTo1UINt32(source,5)
@setBitTo1UINt32(target,4)
new =llvmPassOnesBB(source,target,arr)

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