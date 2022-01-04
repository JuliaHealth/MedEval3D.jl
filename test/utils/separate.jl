
    using Revise, Parameters, Logging, Test
    using CUDA
    includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
    using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
    using Main.BitWiseUtils,Main.ResultListUtils, Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils, Main.ScanForDuplicates
    
    
    for i in 1:100000
        result = CUDA.zeros(1)
        sourceCPU = rand(UInt32,1)[1]
        targetCPU = rand(UInt32,1)[1]
        source= CuArray([sourceCPU])
        target= CuArray([targetCPU])

        function testKernelPassOnes(source,target,result)
            result[1] = @bitPassOnes(source[1],target[1])
            return
        end
        @cuda threads=32 blocks=1 testKernelPassOnes(source,target,result)
        result[1]
    end
    