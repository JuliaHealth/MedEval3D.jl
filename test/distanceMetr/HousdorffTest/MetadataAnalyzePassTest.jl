
using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils, Main.PrepareArrtoBool
using Main.ResultListUtils, Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils, Main.ScanForDuplicates


#########metaDataWarpIter
singleVal = CUDA.zeros(14)

threads=(32,5)
# blocks =2
# mainArrDims= (5,5,5)
# dataBdim = (2,2,2)
blocks =8
mainArrDims= (516,523,421)
dataBdim = (43,21,17)
metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);
#metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,dataBdim),1:9,2:3,4:6,: );
metaDataDims=size(metaData)
loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength = calculateLoopsIter(dataBdim,threads[1],threads[2],metaDataDims,blocks)

function metaDataWarpIterKernel(singleVal,metaDataDims,loopWarpMeta,metaDataLength)

  
    MetadataAnalyzePass.@metaDataWarpIter(metaDataDims,loopWarpMeta,metaDataLength,
    begin
      @ifY 1 if(isInRange) @atomic singleVal[]+=1 end
      #if(linIdexMeta>7000) 
      #@ifY 1 CUDA.@cuprint "linIdexMeta $(linIdexMeta) offset = $(((blockIdxX()-1)*loopWarpMeta*blockDimX()))  xMeta $(xMeta)  yMeta $(yMeta)  zMeta $(zMeta) isInRange $(isInRange) id  idX $(blockIdxX()) \n"   
      #end  
    # @ifXY 1 1    CUDA.@cuprint "linIndex $(linIndex)"
    #CUDA.@cuprint "linIndex $(linIndex) \n "
end)
    
    return
end
@cuda threads=threads blocks=blocks  metaDataWarpIterKernel(singleVal,metaDataDims,loopWarpMeta,metaDataLength)
@test singleVal[1]==metaDataDims[1]*metaDataDims[2]*metaDataDims[3]











####### exOnWarpIfNotFull


using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils, Main.PrepareArrtoBool
using Main.ResultListUtils, Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils, Main.ScanForDuplicates

threads=(32,7)
blocks=(1)
resCounter= CUDA.zeros(Int32,1)
numbOfLoops = 200
function exOnWarpKernelA(resCounter,numbOfLoops)
  for i in 1:numbOfLoops
      @exOnWarp i @ifX 1  @atomic resCounter[1]+=1
      # MetadataAnalyzePass.@exOnWarp i @ifX 1  CUDA.@cuprint "i $(i)  mod $(mod(i,blockDimY() ))  idY   $(threadIdxY()) \n"   #@atomic resCounter[1]+=1
      
  end
  return
end 
    @cuda threads=threads blocks=blocks exOnWarpKernelA(resCounter,numbOfLoops)

    @test resCounter[1]==numbOfLoops





#     using Revise, Parameters, Logging, Test
#     using CUDA
#     includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
#     using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
#     using Main.ResultListUtils, Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils, Main.ScanForDuplicates
    
# threads=(17,13)
# blocks=(1)
# resCounter= CUDA.zeros(Int64,1)
# function exOnWarpIfBoolKernelA(resCounter)
#     #bool true
#   for i in 1:200
#       MetadataAnalyzePass.@exOnWarpIfBool true i @ifX 1 @atomic resCounter[1]+=1
#   end
#   #bool false
#     for i in 1:200
#       MetadataAnalyzePass.@exOnWarpIfBool false i @ifX 1 @atomic resCounter[1]+=1
#   end
#   return
# end 
#     @cuda threads=threads blocks=blocks exOnWarpIfBoolKernelA(resCounter)

#     @test resCounter[1]==200



##################### load counters

using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.ResultListUtils, Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils, Main.ScanForDuplicates

singleVal = CUDA.zeros(14)

threads=(32,4)
blocks =1
mainArrDims= (516,523,826)
dataBdim = (43,21,17)
 metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,dataBdim),1:3,2:5,4:6,: );
#metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,dataBdim),1:9,2:3,4:6,: );
metaDataDims=size(metaData)
loopXMeta= fld(metaDataDims[1],threads[1])
loopYZMeta= fld(metaDataDims[2]*metaDataDims[3],blocks )
loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength = calculateLoopsIter(dataBdim,threads[1],threads[2],metaDataDims,blocks)

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

function loadCountersKernel(loopWarpMeta,metaDataLength,metaData,metaDataDims,loopXMeta,loopYZMeta,shmemSum,globalFpResOffsetCounter, globalFnResOffsetCounter)

  tobeEx= true
  locArr= UInt32(0)
    MetadataAnalyzePass.@metaDataWarpIter(metaDataDims,loopWarpMeta,metaDataLength,
    begin
      MetadataAnalyzePass.@loadCounters()
    #@ifY 1    CUDA.@cuprint "xMeta $(xMeta) yMeta $(xMeta) zMeta $(zMeta)" 
    #CUDA.@cuprint "linIndex $(linIndex) \n "
end)
    
    return
end
@cuda threads=threads blocks=blocks loadCountersKernel(loopWarpMeta,metaDataLength,metaData,metaDataDims,loopXMeta,loopYZMeta,shmemSum,globalFpResOffsetCounter, globalFnResOffsetCounter)
sum(shmemSum)

@test Int64(globalFpResOffsetCounter[1]) ==Int64(ceil((16*1.5)+(17*1.5)))
@test  Int64(globalFnResOffsetCounter[1])==Int64(ceil((18*1.5)+(17*1.5)))


############ analyzeMetadataFirstPass
using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.ResultListUtils, Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils, Main.ScanForDuplicates

threads=(32,4)
blocks =3
mainArrDims= (516,523,826)
dataBdim = (43,21,17)
metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,dataBdim),1:3,2:5,4:6,: );
#metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,dataBdim),1:9,2:3,4:6,: );
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
loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength = calculateLoopsIter(dataBdim,threads[1],threads[2],metaDataDims,blocks)

function analyzeMetadataFirstPassKernel(loopWarpMeta,metaDataLength,workQueaue,workQueaueCounter,metaData,metaDataDims,loopXMeta,loopYZMeta,globalFpResOffsetCounter, globalFnResOffsetCounter)
  shmemSum =  @cuStaticSharedMem(Float32,(35,16)) # we need this additional 33th an 34th spots
  MetadataAnalyzePass.@analyzeMetadataFirstPass()
    
    return
end
@cuda threads=threads blocks=blocks analyzeMetadataFirstPassKernel(loopWarpMeta,metaDataLength,workQueaue,workQueaueCounter,metaData,metaDataDims,loopXMeta,loopYZMeta,globalFpResOffsetCounter, globalFnResOffsetCounter)


# @test length(filter(it-> it>0,Array(workQueaue[:,3])))== 4
@test workQueaueCounter[1]== 4

@test metaData[1,1,1,1]==1
@test metaData[1,1,1,2]==1
@test metaData[1,1,2,2]==0
@test metaData[2,2,2,1]==1
@test metaData[2,2,2,2]==1


first = Int64(metaData[1,1,1,(getResOffsetsBeg()-1)+1])
sec = Int64(metaData[1,1,1,(getResOffsetsBeg()-1)+3])
thir = Int64(metaData[1,1,1,(getResOffsetsBeg()-1)+5])
fourth = Int64(metaData[1,1,1,(getResOffsetsBeg()-1)+7])
fifth = Int64(metaData[1,1,1,(getResOffsetsBeg()-1)+9])
sixth = Int64(metaData[1,1,1,(getResOffsetsBeg()-1)+11])
seventh = Int64(metaData[1,1,1,(getResOffsetsBeg()-1)+13])
@test sec-first >metaData[1,1,1, getBeginingOfFpFNcounts()+1-1]
@test thir-sec>metaData[1,1,1, getBeginingOfFpFNcounts()+3-1]
@test fourth-thir>metaData[1,1,1, getBeginingOfFpFNcounts()+5-1]
@test fifth-fourth>metaData[1,1,1, getBeginingOfFpFNcounts()+7-1]
@test sixth-fifth>metaData[1,1,1, getBeginingOfFpFNcounts()+11-1]
@test seventh-sixth>metaData[1,1,1, getBeginingOfFpFNcounts()+13-1]


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





##################### checkIsActiveOrFullOr and setIsToBeActive
using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils



singleVal = CUDA.zeros(14)
threads=(32,4)
blocks =7
mainArrDims= (516,523,826)
dataBdim = (43,21,17)
metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,dataBdim),1:3,2:4,1:3,: );
metaDataDims=size(metaData)

globalFpResOffsetCounter= CUDA.zeros(UInt32,1)
globalFnResOffsetCounter= CUDA.zeros(UInt32,1)
shmemSum =  CUDA.zeros(Float32,35,16) # we need this additional 33th an 34th 35th spots
workQueaue= WorkQueueUtils.allocateWorkQueue(100,100)
workQueaueCounter= CUDA.zeros(UInt32,1)
# simulating diffrent scenario that should lead to active or inactive 
# blocks and so they should be pushed to work queue or not...
loopXMeta= fld(metaDataDims[1],threads[1])
loopYZMeta= fld(metaDataDims[2]*metaDataDims[3],blocks )

#first already active
xMeta, yMeta, zMeta = 1,0,0
metaData[xMeta,yMeta+1,zMeta+1,getActiveGoldNumb() ]=1
#to be activated
xMeta, yMeta, zMeta = 2,1-1,1-1
metaData[xMeta,yMeta+1,zMeta+1,getIsToBeActivatedInGoldNumb() ]=1
#full should not be active
xMeta, yMeta, zMeta = 2,2-1,1-1
metaData[xMeta,yMeta+1,zMeta+1,getIsToBeActivatedInGoldNumb() ]=1
metaData[xMeta,yMeta+1,zMeta+1,getFullInGoldNumb() ]=1

xMeta, yMeta, zMeta = 2,2-1,2-1
metaData[xMeta,yMeta+1,zMeta+1,getActiveGoldNumb() ]=1
metaData[xMeta,yMeta+1,zMeta+1,getFullInGoldNumb() ]=1

##### now in segm
#first already active
xMeta, yMeta, zMeta = 3,3-1,1-1
metaData[xMeta,yMeta+1,zMeta+1,getActiveSegmNumb() ]=1
#to be activated
xMeta, yMeta, zMeta = 3,1-1,1-1
metaData[xMeta,yMeta+1,zMeta+1,getIsToBeActivatedInSegmNumb() ]=1
#full should not be active
xMeta, yMeta, zMeta = 3,2-1,1-1
metaData[xMeta,yMeta+1,zMeta+1,getIsToBeActivatedInSegmNumb() ]=1
metaData[xMeta,yMeta+1,zMeta+1,getFullInSegmNumb() ]=1

xMeta, yMeta, zMeta = 3,2-1,2-1
metaData[xMeta,yMeta+1,zMeta+1,getActiveSegmNumb() ]=1
metaData[xMeta,yMeta+1,zMeta+1,getFullInSegmNumb() ]=1


loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength = calculateLoopsIter(dataBdim,threads[1],threads[2],metaDataDims,blocks)

#as we have counters we check in shmem are they correct

function checkIsToBeActive(loopWarpMeta,metaDataLength,dataBdim,workQueaueCounter,workQueaue,metaData,metaDataDims,loopXMeta,loopYZMeta,shmemSum,globalFpResOffsetCounter, globalFnResOffsetCounter)
 
  shmemSum =  @cuStaticSharedMem(Float32,(35,16)) # we need this additional 33th an 34th spots
  sourceShmem =  @cuDynamicSharedMem(Bool,(dataBdim[1]+2,dataBdim[2]+2,dataBdim[3]+2)) # we need this additional 33th an 34th spots
  tobeEx= true
  locArr= UInt32(0)
  MetadataAnalyzePass.@metaDataWarpIter( metaDataDims,loopWarpMeta,metaDataLength,
      begin
       MetadataAnalyzePass.@checkIsActiveOrFullOr()
      sync_threads()
      MetadataAnalyzePass.@setIsToBeActive()
      sync_threads()
      #resetting
      @exOnWarp 30 sourceShmem[(threadIdxX()+1)] = false
      @exOnWarp 31 sourceShmem[(threadIdxX()+1)+33]= false
      @exOnWarp 32 sourceShmem[(threadIdxX()+1)+33*2] = false     
      @exOnWarp 33 sourceShmem[(threadIdxX()+1)+33*3] = false
      @exOnWarp 34 sourceShmem[(threadIdxX()+1)+33*4] = false
      @exOnWarp 35 sourceShmem[(threadIdxX()+1)+33*5] = false
    end)
    
    return
end
@cuda threads=threads blocks=blocks checkIsToBeActive(loopWarpMeta,metaDataLength,dataBdim,workQueaueCounter,workQueaue,metaData,metaDataDims,loopXMeta,loopYZMeta,shmemSum,globalFpResOffsetCounter, globalFnResOffsetCounter)
workQueaue[1,:]
Int64(sum(workQueaue))
Int64(workQueaueCounter[1])
ss = 0
for i in 1:length(workQueaue)
  if( workQueaue[i,1]>0 )
    ss+=1
  end
end
ss


function testForPresenceInWorkQueue(arr)::Bool
  outBool = false
  for i in 1:size(workQueaue)[2]
    if(workQueaue[:,i]==arr )
      outBool=true
    end  
  end  
return outBool
end

using Logging
for i in 1:10
  @info "$(Int64.(workQueaue[:,i])) \n"
end

metaData[1,1,1,getFullInGoldNumb() ]
workQueaue

@test  testForPresenceInWorkQueue([0,1,1,1])
@test  testForPresenceInWorkQueue([1,1,1,1])

@test  !testForPresenceInWorkQueue([1,2,1,1])#full
@test  !testForPresenceInWorkQueue([1,2,2,1])#full

@test  testForPresenceInWorkQueue([2,3,1,0])
@test  testForPresenceInWorkQueue([2,1,1,0])

@test  !testForPresenceInWorkQueue([2,2,1,0])#full
@test  !testForPresenceInWorkQueue([2,2,2,0])#full



@test metaData[xMeta,yMeta+1,zMeta+1,getFullInGoldNumb() ]==1


# @test Int64(globalFpResOffsetCounter[1]) ==Int64(ceil((16*1.5)+(17*1.5)))
# @test  Int64(globalFnResOffsetCounter[1])==Int64(ceil((18*1.5)+(17*1.5)))
