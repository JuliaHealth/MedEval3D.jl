
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

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\MetadataAnalyzePass.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils
using Main.MetadataAnalyzePass


#########metaDataWarpIter
singleVal = CUDA.zeros(14)

threads=(32,5)
blocks =8
mainArrDims= (516,523,826)
datBdim = (43,21,17)
 metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,datBdim),1:17,2:18,4:10,: );
#metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,datBdim),1:9,2:3,4:6,: );
metaDataDims=size(metaData)
loopXMeta= fld(metaDataDims[1],threads[1])
loopYZMeta= fld(metaDataDims[2]*metaDataDims[3],blocks )

function metaDataWarpIterKernel(singleVal,metaDataDims,loopXMeta,loopYZMeta)

  
    MetadataAnalyzePass.@metaDataWarpIter( metaDataDims,loopXMeta,loopYZMeta,
    begin
      @ifY 1 @atomic singleVal[]+=1
      #@ifX 1 CUDA.@cuprint "   xMeta $(xMeta)  yMeta $(yMeta)  zMeta $(zMeta) id  idX $(blockIdxX()) \n"   

    # @ifXY 1 1    CUDA.@cuprint "linIndex $(linIndex)"
    #CUDA.@cuprint "linIndex $(linIndex) \n "
end)
    
    return
end
@cuda threads=threads blocks=blocks metaDataWarpIterKernel(singleVal,metaDataDims,loopXMeta,loopYZMeta)
@test singleVal[1]==metaDataDims[1]*metaDataDims[2]*metaDataDims[3]











####### exOnWarpIfNotFull





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

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\MetadataAnalyzePass.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils
using Main.MetadataAnalyzePass

threads=(32,7)
blocks=(1)
resCounter= CUDA.zeros(Int32,1)
numbOfLoops = 200
function exOnWarpKernelA(resCounter,numbOfLoops)
  for i in 1:numbOfLoops
      MetadataAnalyzePass.@exOnWarp i @ifX 1  @atomic resCounter[1]+=1
      # MetadataAnalyzePass.@exOnWarp i @ifX 1  CUDA.@cuprint "i $(i)  mod $(mod(i,blockDimY() ))  idY   $(threadIdxY()) \n"   #@atomic resCounter[1]+=1
      
  end
  return
end 
    @cuda threads=threads blocks=blocks exOnWarpKernelA(resCounter,numbOfLoops)

    @test resCounter[1]==numbOfLoops





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

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\MetadataAnalyzePass.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils
using Main.MetadataAnalyzePass

threads=(17,13)
blocks=(1)
resCounter= CUDA.zeros(Int64,1)
function exOnWarpIfBoolKernelA(resCounter)
    #bool true
  for i in 1:200
      MetadataAnalyzePass.@exOnWarpIfBool true i @ifX 1 @atomic resCounter[1]+=1
  end
  #bool false
    for i in 1:200
      MetadataAnalyzePass.@exOnWarpIfBool false i @ifX 1 @atomic resCounter[1]+=1
  end
  return
end 
    @cuda threads=threads blocks=blocks exOnWarpIfBoolKernelA(resCounter)

    @test resCounter[1]==200



##################### load counters


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

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\MetadataAnalyzePass.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.MetadataAnalyzePass,Main.MetaDataUtils


#########loadCounters
singleVal = CUDA.zeros(14)

threads=(32,4)
blocks =1
mainArrDims= (516,523,826)
datBdim = (43,21,17)
 metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,datBdim),1:3,2:5,4:6,: );
#metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,datBdim),1:9,2:3,4:6,: );
metaDataDims=size(metaData)
loopXMeta= fld(metaDataDims[1],threads[1])
loopYZMeta= fld(metaDataDims[2]*metaDataDims[3],blocks )

globalFpResOffsetCounter= CUDA.zeros(UInt32,1)
globalFnResOffsetCounter= CUDA.zeros(UInt32,1)

shmemSum =  CUDA.zeros(Float32,35,16) # we need this additional 33th an 34th 35th spots

# setting some counters in blocks of metadata that are on diagonal

for j in 1:2 
  ii = 0
  for i in 1:14
    ii+=i
    metaData[j,j,j, getBeginingOfFpFNcounts()+i]=i
  end
  metaData[j,j,j, getBeginingOfFpFNcounts()+16]=15+j
  metaData[j,j,j, getBeginingOfFpFNcounts()+17]=16+j
end  

#as we have counters we check in shmem are they correct

function loadCountersKernel(metaData,metaDataDims,loopXMeta,loopYZMeta,shmemSum,globalFpResOffsetCounter, globalFnResOffsetCounter)

  tobeEx= true
  locArr= UInt32(0)
    MetadataAnalyzePass.@metaDataWarpIter(metaDataDims,loopXMeta,loopYZMeta,
    begin
      MetadataAnalyzePass.@loadCounters(tobeEx,locArr)
    #@ifY 1    CUDA.@cuprint "xMeta $(xMeta) yMeta $(xMeta) zMeta $(zMeta)" 
    #CUDA.@cuprint "linIndex $(linIndex) \n "
end)
    
    return
end
@cuda threads=threads blocks=blocks loadCountersKernel(metaData,metaDataDims,loopXMeta,loopYZMeta,shmemSum,globalFpResOffsetCounter, globalFnResOffsetCounter)
sum(shmemSum)

@test Int64(globalFpResOffsetCounter[1]) ==Int64(ceil((16*1.5)+(17*1.5)))
@test  Int64(globalFnResOffsetCounter[1])==Int64(ceil((18*1.5)+(17*1.5)))


############ analyzeMetadataFirstPass
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

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\MetadataAnalyzePass.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils

threads=(32,4)
blocks =3
mainArrDims= (516,523,826)
datBdim = (43,21,17)
 metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,datBdim),1:3,2:5,4:6,: );
#metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,datBdim),1:9,2:3,4:6,: );
metaDataDims=size(metaData)
loopXMeta= fld(metaDataDims[1],threads[1])
loopYZMeta= fld(metaDataDims[2]*metaDataDims[3],blocks )
fpTotal=150
fnTotal=125
globalFpResOffsetCounter= CUDA.zeros(UInt32,1)
globalFnResOffsetCounter= CUDA.zeros(UInt32,1)

workQueaue= WorkQueueUtils.allocateWorkQueue(fpTotal,fnTotal)

metaData[2,2,2,2]=UInt32(1)

workQueaueCounter= CUDA.zeros(UInt32,1)

shmemSum =  CUDA.zeros(Float32,35,16) # we need this additional 33th an 34th 35th spots

for j in 1:2 
  ii = 0
  for i in 1:14
    ii+=i
    metaData[j,j,j, getBeginingOfFpFNcounts()+i]=i
  end
  metaData[j,j,j, getBeginingOfFpFNcounts()+16]=15+j
  metaData[j,j,j, getBeginingOfFpFNcounts()+17]=16+j
end  

function analyzeMetadataFirstPassKernel(workQueaue,workQueaueCounter,metaData,metaDataDims,loopXMeta,loopYZMeta,globalFpResOffsetCounter, globalFnResOffsetCounter)
  shmemSum =  @cuStaticSharedMem(Float32,(35,16)) # we need this additional 33th an 34th spots
  MetadataAnalyzePass.@analyzeMetadataFirstPass()
    
    return
end
@cuda threads=threads blocks=blocks analyzeMetadataFirstPassKernel(workQueaue,workQueaueCounter,metaData,metaDataDims,loopXMeta,loopYZMeta,globalFpResOffsetCounter, globalFnResOffsetCounter)
#check are all required blocks in the work queue

Int64(sum(workQueaue))
Int64(workQueaueCounter[1])
Int64.(workQueaue[1,:])

maximum(sort(Array(workQueaue[:,1]))) ==metaDataDims[1]
maximum(sort(Array(workQueaue[:,2]))) ==metaDataDims[2]
maximum(sort(Array(workQueaue[:,3]))) ==metaDataDims[3]

length(filter(it-> it>0,Array(workQueaue[:,3])))== metaDataDims[1]*metaDataDims[2]*metaDataDims[3]*2

Int64(sum(workQueaue))
workQueaue[1,1]==1 
workQueaue[1,1]==1 
workQueaue[1,2]==1 
workQueaue[1,3]==1 
#workQueaue[1,4]==0 

workQueaue[2,1]==1 
workQueaue[2,2]==1 
workQueaue[2,3]==1 
#workQueaue[2,4]==1

workQueaue[3,2]== 2
workQueaue[3,2]==2
workQueaue[3,2]==2 
#workQueaue[3,4]==0 
metaData[2,2,2,2]
#check if offsets are calculated correctly