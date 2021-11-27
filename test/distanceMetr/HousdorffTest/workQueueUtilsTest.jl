using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils
using Main.MainLoopKernel, Main.WorkQueueUtils, Main.Housdorff

mainArrDims= (67,177,90);
dataBdim = (43,21,17);
robustnessPercent= 0.9;
numberToLooFor= 2;
goldGPU,segmGPU= CUDA.zeros(mainArrDims),CUDA.zeros(mainArrDims);

boolKernelArgs, mainKernelArgs,threadsBoolKern,blocksBoolKern ,threadsMainKern,blocksMainKern ,shmemSizeBool,shmemSizeMain= preparehousedorfKernel(goldGPU,segmGPU,robustnessPercent,numberToLooFor)

goldGPUa,segmGPUa,dilatationArrsA,dilatationArrsB, mainArrDims,dataBdim ,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent   ,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter   ,workQueaueCounter,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount   ,globalIterationNumb,workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter    ,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter   ,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter   ,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter   ,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed   ,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY   ,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp,resList = mainKernelArgs


function addToWorkQueueKernel(workQueueFp,workQueaueAcounterFp, metaX,metaY,metaZ,isGold)
       goldGPUa,segmGPUa,dilatationArrsA,dilatationArrsB, mainArrDims,dataBdim ,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent   ,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter   ,workQueaueCounter,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount   ,globalIterationNumb,workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter    ,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter   ,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter   ,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter   ,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed   ,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY   ,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp,resList = mainKernelArgs

      @ifXY 1 1 WorkQueueUtils.appendToWorkQueue(workQueueFp,workQueaueAcounterFp, metaX,metaY,metaZ,isGold )
      return
  end
  @cuda threads=threads blocks=blocks addToWorkQueueKernel(workQueueFp,workQueaueAcounterFp, metaX,metaY,metaZ,isGold)

  @cuda threads=threads blocks=blocks addToWorkQueueKernel(workQueueFp,workQueaueAcounterFp, metaX,metaY,metaZ,isGold)


  @test workQueaueAcounterFp[1]==2
  @test   workQueueFp[1,1]== UInt8(metaX)
  @test   workQueueFp[1,2]== UInt8(metaY)
   @test  workQueueFp[1,3]== UInt8(metaZ)
   @test  workQueueFp[1,4]== UInt8(isGold)

   @test  workQueueFp[2,1]== UInt8(metaX)
   @test  workQueueFp[2,2]== UInt8(metaY)
   @test  workQueueFp[2,3]== UInt8(metaZ)
   @test  workQueueFp[2,4]== UInt8(isGold)

   @test  workQueueFp[3,1]== UInt8(0)
   @test  workQueueFp[3,2]== UInt8(0)
   @test  workQueueFp[3,3]== UInt8(0)
   @test  workQueueFp[3,4]== UInt8(0)
  old= UInt8(1.0)
  workQueueFp[old,1]= UInt8(metaX)