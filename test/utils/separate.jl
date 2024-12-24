
    using Revise, Parameters, Logging, Test
    using CUDA
    includet("./test/includeAllUseFullForTest.jl")
    using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
    using ..BitWiseUtils,..ResultListUtils, ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils, ..ScanForDuplicates
    
    
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
    