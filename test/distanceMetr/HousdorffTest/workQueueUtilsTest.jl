using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils
using Main.MainLoopKernel, Main.WorkQueueUtils, Main.Housdorff
using CUDA, Logging,Main.CUDAGpuUtils, Main.ResultListUtils,Main.WorkQueueUtils,Main.ScanForDuplicates, Logging,StaticArrays, Main.IterationUtils, Main.ReductionUtils, Main.CUDAAtomicUtils,Main.MetaDataUtils

mainArrDims= (67,177,90);
dataBdim = (32,32,32);
robustnessPercent= 0.9;
numberToLooFor= 2;
goldGPU,segmGPU= CUDA.zeros(mainArrDims),CUDA.zeros(mainArrDims);

boolKernelArgs, mainKernelArgs,threadsBoolKern,blocksBoolKern ,threadsMainKern,blocksMainKern ,shmemSizeBool,shmemSizeMain= preparehousedorfKernel(goldGPU,segmGPU,robustnessPercent,numberToLooFor)

dilatationArrsA,dilatationArrsB, mainArrDims,dataBdim ,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent   ,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter   ,workQueaueCounter,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount   ,globalIterationNumb,workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter    ,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter   ,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter   ,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter   ,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed   ,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY   ,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp,resList = mainKernelArgs


function addToWorkQueueKernel(dilatationArrsA,dilatationArrsB, mainArrDims,dataBdim ,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent   ,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter   ,workQueaueCounter,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount   ,globalIterationNumb,workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter    ,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter   ,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter   ,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter   ,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed   ,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY   ,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp,resList)

    if (blockIdxX()==1)
      @ifXY 1 1 WorkQueueUtils.@appendToWorkQueue(1,1,1,0)
      @ifXY 2 1 WorkQueueUtils.@appendToWorkQueue(2,1,1,0)
      @ifXY 3 1 WorkQueueUtils.@appendToWorkQueue(1,2,1,0)
      @ifXY 4 1 WorkQueueUtils.@appendToWorkQueue(1,1,2,0)
      @ifXY 5 1 WorkQueueUtils.@appendToWorkQueue(1,2,2,0)
      @ifXY 6 1 WorkQueueUtils.@appendToWorkQueue(2,1,2,0)
      @ifXY 7 1 WorkQueueUtils.@appendToWorkQueue(2,2,2,0)
      @ifXY 8 1 WorkQueueUtils.@appendToWorkQueue(2,2,1,0)
    end   
      return
  end
  @cuda threads=(32,32) blocks=20 addToWorkQueueKernel(dilatationArrsA,dilatationArrsB, mainArrDims,dataBdim ,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent   ,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter   ,workQueaueCounter,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount   ,globalIterationNumb,workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter    ,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter   ,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter   ,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter   ,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed   ,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY   ,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp,resList)


  
  @test workQueueEEEcounter[1]==1
  @test workQueueEEOcounter[1]==1
  @test workQueueEOEcounter[1]==1
  @test workQueueOEEcounter[1]==1
  @test workQueueOOEcounter[1]==1
  @test workQueueEOOcounter[1]==1
  @test workQueueOEOcounter[1]==1
  @test workQueueOOOcounter[1]==1

Int64(workQueueEEOcounter[1])
Int64(workQueueEOOcounter[1])
