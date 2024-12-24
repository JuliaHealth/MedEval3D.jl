using Revise, Parameters, Logging, Test
using CUDA
includet("./test/includeAllUseFullForTest.jl")
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
using ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils
using ..MainLoopKernel, ..WorkQueueUtils, ..Housdorff
using CUDA, Logging,..CUDAGpuUtils, ..ResultListUtils,..WorkQueueUtils,..ScanForDuplicates, Logging,StaticArrays, ..IterationUtils, ..ReductionUtils, ..CUDAAtomicUtils,..MetaDataUtils

mainArrDims= (67,177,90);
dataBdim = (32,32,32);
robustnessPercent= 0.9;
numberToLooFor= 2;
goldGPU,segmGPU= CUDA.zeros(mainArrDims),CUDA.zeros(mainArrDims);

boolKernelArgs, mainKernelArgs,threadsBoolKern,blocksBoolKern ,threadsMainKern,blocksMainKern ,shmemSizeBool,shmemSizeMain= preparehousedorfKernel(goldGPU,segmGPU,robustnessPercent,numberToLooFor)



function addToWorkQueueKernel(dilatationArrsA,dilatationArrsB, mainArrDims,dataBdim ,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent   ,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter   ,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount   ,globalIterationNumb,workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter    ,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter   ,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter   ,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter   ,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed   ,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY   ,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp,resList,inBlockLoopXZIterWithPadding,paddingStore,shmemblockDataLenght,shmemblockDataLoop)

    if (blockIdxX()==1)
      @ifXY 1 1 WorkQueueUtils.@appendToWorkQueue(1,1,1,0)

    end   
      return
  end
  @cuda threads=(32,32) blocks=20 addToWorkQueueKernel(dilatationArrsA,dilatationArrsB, mainArrDims,dataBdim ,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent   ,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter   ,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount   ,globalIterationNumb,workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter    ,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter   ,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter   ,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter   ,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed   ,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY   ,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp,resList,inBlockLoopXZIterWithPadding,paddingStore,shmemblockDataLenght,shmemblockDataLoop)


  
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
