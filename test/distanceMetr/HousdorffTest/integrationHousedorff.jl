    ######### getting all together in Housedorff 

    using Revise, Parameters, Logging, Test
    using CUDA
    includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
    using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
    using Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils,Main.ResultListUtils, Main.Housdorff


    mainArrDims= (60,60,60);
    mainArrCPU= ones(UInt8,mainArrDims);
    refArrCPU = ones(UInt8,mainArrDims);
    ##### we will create two planes 20 units apart from each 
    mainArrCPU[10:50,10:50,10].= true;
    refArrCPU[10:50,10:50,30].= true;


        
    goldGPU = CuArray(mainArrCPU);
    segmGPU= CuArray(mainArrCPU);

    robustnessPercent= 0.95
    numberToLooFor=1
    boolKernelArgs, mainKernelArgs,threadsBoolKern,blocksBoolKern ,threadsMainKern,blocksMainKern= Housdorff.preparehousedorfKernel(goldGPU,segmGPU,robustnessPercent,numberToLooFor);
    threadsBoolKern,Int64(blocksBoolKern) ,threadsMainKern,Int64(blocksMainKern)

    Housdorff.getHousedorffDistance(goldGPU,segmGPU,boolKernelArgs,mainKernelArgs,threadsBoolKern,blocksBoolKern ,threadsMainKern,blocksMainKern)

