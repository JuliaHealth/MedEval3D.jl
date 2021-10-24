using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAAtomicUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MetaDataUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\WorkQueueUtils.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.WorkQueueUtils,Main.MetaDataUtils

fpTotal=150
fnTotal=125
metaX= 2
metaY= 3
metaZ= 4
isGold = true
threads=(32,4)
blocks =1

workQueueFp= WorkQueueUtils.allocateWork_Fp_Fn_Queues(fpTotal,fnTotal)
workQueaueAcounterFp = CUDA.zeros(UInt32,1)

function addToWorkQueueKernel(workQueueFp,workQueaueAcounterFp, metaX,metaY,metaZ,isGold)

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