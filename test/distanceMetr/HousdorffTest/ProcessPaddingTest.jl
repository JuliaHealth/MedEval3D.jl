
# using  Test, Revise,CUDA 
# includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")

# includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
# includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\mainHouseDorffKernel\\HFUtils.jl")
# includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\mainHouseDorffKernel\\ProcessPadding.jl")
# includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\mainHouseDorffKernel\\ProcessMainData.jl")
# includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\GPUtestUtils.jl")

# using Main.CUDAGpuUtils, Main.HFUtils,Cthulhu


# using Main.CUDAGpuUtils, Main.HFUtils,Cthulhu

# #@testset "processPaddingTest" begin 

#     testArrInCPU= falses(100,100,100);
#     #testArrInCPU[CartesianIndex(1,1,1)]= true;
#     testArrInCPU[CartesianIndex(1,1,1)]= true
#     testArrInCPU[CartesianIndex(5,5,5)]= true
#     testArrIn = CuArray(testArrInCPU);

#     metaDataCPU = falses(15,15,15,4); # all neighbouring set to be inactive and not full
#     metaData= CuArray(metaDataCPU);
#     debugArr= CUDA.zeros(Bool,9);
#     resArraysCounter=CUDA.zeros(Int32,1);


#     referenceArray= CUDA.ones(Bool,100,100,100);
#     resArray = CUDA.zeros(UInt16,50,50,50);
#     blockBeginingX,blockBeginingY,blockBeginingZ =UInt8(33),UInt8(33),UInt8(33);
#     currBlockX,currBlockY,currBlockZ =UInt8(2),UInt8(2),UInt8(2);
#     isPassGold = true;
#     #metaDataCPU[2,2,3,:].= true

#     metadataDims = size(metaData);
#     mainQuesCounter=CUDA.zeros(Int32,1);
#     mainWorkQueue = CUDA.zeros(UInt8,10,4);#simulating place for 10 blocks
#     iterationNumber= UInt32(2);
# function testKernelForPaddingAnalysis(analyzedArr, refAray,blockBeginingX,blockBeginingY,blockBeginingZ,resArray,resArraysCounter
#                                     ,isPassGold,currBlockX,currBlockY,currBlockZ,metaData,metadataDims,mainQuesCounter,mainWorkQueue,iterationNumber,debugArr)
#         resShmem =  @cuStaticSharedMem(Bool,(34,34,34))#+2 in order to get the one padding 
#         ProcessMainData.executeDataIterFirstPassWithPadding(analyzedArr, refAray,blockBeginingX,blockBeginingY,blockBeginingZ,resShmem,resArray,resArraysCounter,currBlockX,currBlockY,currBlockZ,isPassGold,metaData,metadataDims,mainQuesCounter,mainWorkQueue,iterationNumber,debugArr)
#         #clearMainShmem(shmem)
#         #now we need to deal with padding in shmem res
#         # processAllPaddingPlanes(blockBeginingX,blockBeginingY,blockBeginingZ,resShmem
#         #                     ,currBlockX,currBlockY,currBlockZ
#         #                     ,analyzedArr,refAray,resArray
#         #                     ,metaData,metadataDims
#         #                     ,isPassGold)


#         return
# end


# @cuda threads=(16,16) blocks=1 testKernelForPaddingAnalysis(testArrIn,referenceArray,blockBeginingX,blockBeginingY,blockBeginingZ,  resArray,
#   resArraysCounter,isPassGold,currBlockX,currBlockY,currBlockZ,metaData,metadataDims,mainQuesCounter,mainWorkQueue,iterationNumber,debugArr) 
#   resArraysCounter[1]
#   Int64(maximum(mainWorkQueue))



#   indicies = CartesianIndices(mainWorkQueue);
#   filtered = filter(ind-> mainWorkQueue[ind]>0,indicies  )


# debugArr[5]
# Int64(mainWorkQueue[2,4])

# Int64(maximum(mainWorkQueue))

# indicies = CartesianIndices(mainWorkQueue);
# filtered = filter(ind-> mainWorkQueue[ind]>0,indicies  )

# metaData[2, 2, 3, 2]==true


# debugArr[1]
# debugArr[2]
# debugArr[3]
# debugArr[4]
# debugArr[5]
# debugArr[6]











# testArrInCPU= falses(100,100,100);
# #testArrInCPU[CartesianIndex(1,1,1)]= true;
# testArrInCPU[CartesianIndex(5,5,5)]= true;
# testArrInCPU[5,5,32]=true

# testArrIn = CuArray(testArrInCPU);
# referenceArray= CUDA.ones(Bool,100,100,100);
# resArray = CUDA.zeros(UInt32,32,32,32);
# blockBeginingX,blockBeginingY,blockBeginingZ =UInt8(0),UInt8(0),UInt8(0);
# currBlockX,currBlockY,currBlockZ =UInt8(2),UInt8(2),UInt8(2);
# isPassGold = true;
# metaDataCPU = falses(15,15,15,4); # all neighbouring set to be inactive and not full
# #metaDataCPU[2,2,3,:].= true

# metaData= CuArray(metaDataCPU);
# metadataDims = size(metaData);
# mainQuesCounter=CUDA.zeros(Int32,1);
# mainWorkQueue = CUDA.zeros(Bool,10,4);#simulating place for 10 blocks
# iterationNumber= UInt32(2);

# @device_code_warntype interactive=true @cuda testKernelForPaddingAnalysis(testArrIn,referenceArray,blockBeginingX,blockBeginingY,blockBeginingZ,  resArray,resArraysCounter,isPassGold,currBlockX,currBlockY,currBlockZ,metaData,metadataDims,mainQuesCounter,mainWorkQueue,iterationNumber,debugArr)

# #end




###########loadMainValues
using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.BitWiseUtils,Main.ResultListUtils, Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils, Main.ScanForDuplicates


mainArrGPU= CUDA.zeros(UInt32, 50,50,20)

dataBdim= (32,10,32)

xMeta,yMeta,zMeta = 1,1,1

resShmemblockData= CUDA.zeros(UInt32, 32,10);
shmemblockData= CUDA.zeros(UInt32, 32,10);
shmemPaddings= CUDA.zeros(Bool, 32,32,6);

rowOne = 0
@setBitTo(rowOne,1,true)
@setBitTo(rowOne,5,true)
@setBitTo(rowOne,32,true)



mainArrGPU[1,1,1]= rowOne
mainArrGPU[1,10,1]= rowOne
mainArrGPU[32,1,1]= rowOne


function testLoadMainValues(shmemPaddings,shmemblockData,resShmemblockData,dataBdim,mainArrGPU,xMeta,yMeta,zMeta)

  sync_threads()


  @loadMainValues(mainArrGPU,xMeta,yMeta,zMeta)

    return
end

@cuda threads=(32,10) blocks=1 testLoadMainValues(shmemPaddings,shmemblockData,resShmemblockData,dataBdim,mainArrGPU,xMeta,yMeta,zMeta)

@test shmemblockData[1,1]==rowOne
@test shmemblockData[1,10]==rowOne
@test shmemblockData[32,1]==rowOne



@test isBit1AtPos(resShmemblockData[1,1],1)
@test isBit1AtPos(resShmemblockData[1,1],2)
@test !isBit1AtPos(resShmemblockData[1,1],3)
@test isBit1AtPos(resShmemblockData[1,1],4)
@test isBit1AtPos(resShmemblockData[1,1],5)
@test isBit1AtPos(resShmemblockData[1,1],6)
@test isBit1AtPos(resShmemblockData[1,1],31)
@test isBit1AtPos(resShmemblockData[1,1],32)

@test isBit1AtPos(resShmemblockData[1,10],1)
@test isBit1AtPos(resShmemblockData[1,10],2)
@test isBit1AtPos(resShmemblockData[1,10],4)
@test isBit1AtPos(resShmemblockData[1,10],5)
@test isBit1AtPos(resShmemblockData[1,10],6)
@test isBit1AtPos(resShmemblockData[1,10],31)
@test isBit1AtPos(resShmemblockData[1,10],32)

@test isBit1AtPos(resShmemblockData[32,1],1)
@test isBit1AtPos(resShmemblockData[32,1],2)
@test isBit1AtPos(resShmemblockData[32,1],4)
@test isBit1AtPos(resShmemblockData[32,1],5)
@test isBit1AtPos(resShmemblockData[32,1],6)
@test isBit1AtPos(resShmemblockData[32,1],31)
@test isBit1AtPos(resShmemblockData[32,1],32)

# holding data about result top, bottom, left right , anterior, posterior ,  paddings

@test shmemPaddings[1,1,1]
@test shmemPaddings[1,10,1]
@test shmemPaddings[32,1,1]

@test shmemPaddings[1,1,2]
@test shmemPaddings[1,10,2]
@test shmemPaddings[32,1,2]

@test shmemPaddings[1,1,3]
@test shmemPaddings[5,1,3]
@test shmemPaddings[32,1,3]

@test shmemPaddings[1,10,3]
@test shmemPaddings[5,10,3]
@test shmemPaddings[32,10,3]

@test shmemPaddings[1,1,4]
@test shmemPaddings[5,1,4]
@test shmemPaddings[32,1,4]
@test !shmemPaddings[32,2,4]


@test shmemPaddings[1,1,5]
@test shmemPaddings[1,5,5]
@test shmemPaddings[1,32,5]


@test shmemPaddings[1,1,6]
@test shmemPaddings[1,5,6]
@test shmemPaddings[1,32,6]

#############  validateData
# using Revise, Parameters, Logging, Test
# using CUDA
# includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
# using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
# using Main.BitWiseUtils,Main.ResultListUtils, Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils, Main.ScanForDuplicates




# mainArr= CUDA.zeros(UInt32, 50,50,10)
# refArr= CUDA.zeros(UInt32, 50,50,10)
# targetArr= CUDA.zeros(UInt32, 50,50,10)

# dataBdim= (32,10,32)

# xMeta,yMeta,zMeta = 1,1,1

# resShmemblockData= CUDA.zeros(UInt32, 32,10);
# shmemblockData= CUDA.zeros(UInt32, 32,10);
# shmemPaddings= CUDA.zeros(Bool, 32,32,6);

# threads=(32,10)
# rowOne = 0
# @setBitTo(rowOne,1,true)
# @setBitTo(rowOne,5,true)
# @setBitTo(rowOne,32,true)

# mainArr[1,1,1]= rowOne
# mainArr[1,10,1]= rowOne
# mainArr[32,1,1]= rowOne

# rowB = 0
# @setBitTo(rowB,1,true)
# @setBitTo(rowB,2,true)
# @setBitTo(rowB,5,true)
# @setBitTo(rowB,6,true)
# @setBitTo(rowB,32,true)

# targetArr[1,10,1]= rowB




# blocks =1
# mainArrDims= (50,50,320)
# metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim)
# #metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,dataBdim),1:9,2:3,4:6,: );
# metaDataDims=size(metaData)
# metaSize= metaDataDims
# workQueue,workQueueCounter= WorkQueueUtils.allocateWorkQueue( max(length(metaData),1) )
# metaData[2,2,2,2]=UInt32(1)
# #setting offsets in metadata
# for i in 1:14
#   metaData[xMeta,yMeta,zMeta,getResOffsetsBeg()+i]=i*10
# end
# isGold = 1
# iterNumb = 1
# resList = allocateResultLists(1000,1000)
# paddingStore = CUDA.zeros(UInt8, metaSize[1],metaSize[2],metaSize[3],32,32)
# inBlockLoopXZIterWithPadding = cld(32,10)
# numberToLooFor = 2
# function testvalidateData(numberToLooFor,inBlockLoopXZIterWithPadding,paddingStore,resList,shmemPaddings,shmemblockData,resShmemblockData,metaData,metaDataDims,mainArrDims,isGold,xMeta,yMeta,zMeta,iterNumb,mainArr,refArr,targetArr,dataBdim,workQueue,workQueueCounter)
#   isMaskFull = true
#   @loadMainValues(mainArr,xMeta,yMeta,zMeta)

#   sync_threads()
#   @validateData(isGold,xMeta,yMeta,zMeta,iterNumb,mainArr,refArr,targetArr)

#     return
# end

# @cuda threads=threads blocks=blocks testvalidateData(inBlockLoopXZIterWithPadding,paddingStore,resList,shmemPaddings,shmemblockData,resShmemblockData,metaData,metaDataDims,mainArrDims,isGold,xMeta,yMeta,zMeta,iterNumb,mainArr,refArr,targetArr,dataBdim,workQueue,workQueueCounter)

# @test shmemPaddings[1,1,4]
# @test shmemPaddings[5,1,4]
# @test shmemPaddings[32,1,4]
# @test !shmemPaddings[32,2,4]

# @test !shmemPaddings[32,2,4]

# @setBitTo(rowB,1,true)
# @setBitTo(rowB,2,true)
# @setBitTo(rowB,5,true)
# @setBitTo(rowB,6,true)
# @setBitTo(rowB,32,true)

# targetArr[1,10,1]= rowB








################## execute Data Iter With Padding  first data 

using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.Housdorff,Main.ProcessMainDataVerB,Main.MainLoopKernel,Main.BitWiseUtils,Main.ResultListUtils, Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils, Main.ScanForDuplicates

mainArr= CUDA.zeros(UInt32, 300,300,10)
refArr= CUDA.zeros(UInt8, 300,300,320)
numbToLooFor = 2
dataBdim= (32,5,32)

############ some arbitrary data  meta 2,2,2 with some results to be set 
xMeta,yMeta,zMeta = 2,2,2


threads=(dataBdim[1],dataBdim[2])
rowOne = UInt32(0)

#below we will organize it so we should have results in all neghbouring blocks - also from corners 
@setBitTo(rowOne,1,true)
@setBitTo(rowOne,5,true)
@setBitTo(rowOne,32,true)

mainArr[33,dataBdim[2]+1,2]= rowOne
mainArr[33,dataBdim[2]*2,2]= rowOne
mainArr[64,dataBdim[2]+1,2]= rowOne
mainArr[64,dataBdim[2]*2,2]= rowOne

mainArr[64,dataBdim[2]+1,2]= rowOne

################   here we will get a block that will become full after the dilatation meta 3,3,3
fullOnesTobecome = UInt32(0)
for bitPos in 1:32
  if(isodd(bitPos))
    @setBitTo(fullOnesTobecome,bitPos,true)
  end
end
for xx in ((2*32)+1):((3*32)), yy in ((2*dataBdim[2])+1):((3*dataBdim[2]))
  mainArr[xx,yy,3]= fullOnesTobecome
end

mainArr[100,16,3]

for xx in ((3*32)+1):((4*32)), yy in ((3*dataBdim[2])+1):((4*dataBdim[2])), zz in ((3*32)+1):((4*32))
  refArr[xx,yy,zz]= 2
end

refArr[33,dataBdim[2]+1,33]= 2
refArr[33,dataBdim[2]*2,34]= 2
refArr[64,dataBdim[2]+1,37]= 2
refArr[64,dataBdim[2]+1,38]= 2


refArr[33,dataBdim[2]*2,34]= 2 #from bottom 
refArr[64,dataBdim[2]+1,63]= 2 #from top
refArr[64-1,dataBdim[2]+1,64]= 2#from right
refArr[64+1,dataBdim[2]+1,64]= 2#from left

refArr[64,dataBdim[2]+2,64]= 2#from posterior
refArr[64,dataBdim[2],64]= 2#from anterior


#### single points to check dilatations
mainArr[130,22,1]
(dataBdim[1]*4)+2
(dataBdim[2]*4)+2
mainArr[(dataBdim[2]*4)+2 ,(dataBdim[2]*4)+2,1]
#top 5,5,1
roww = UInt32(0)
@setBitTo(roww,1,true)
mainArr[(dataBdim[1]*4)+2 ,(dataBdim[2]*4)+2,1]= roww
#bottom 5,5,2
roww = UInt32(0)
@setBitTo(roww,32,true)
mainArr[(dataBdim[1]*4)+2 ,(dataBdim[2]*4)+2,2]= roww
#left 5,5,3
roww = UInt32(0)
@setBitTo(roww,5,true)
mainArr[(dataBdim[1]*4)+1 ,(dataBdim[2]*4)+2,3]= roww
#right 5,5,4
roww = UInt32(0)
@setBitTo(roww,5,true)
mainArr[(dataBdim[1]*5) ,(dataBdim[2]*4)+2,4]= roww
#anterior 5,5,5
roww = UInt32(0)
@setBitTo(roww,5,true)
mainArr[(dataBdim[1]*4)+2 ,(dataBdim[2]*4)+1,5]= roww
#posterior 5,5,6
roww = UInt32(0)
@setBitTo(roww,5,true)
mainArr[(dataBdim[1]*4)+2 ,(dataBdim[2]*5),6]= roww


### configurations

mainArrDims= size(mainArr)
metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim)
#metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,dataBdim),1:9,2:3,4:6,: );
metaDataDims=size(metaData)

workQueaue,workQueaueCounter= WorkQueueUtils.allocateWorkQueue( max(length(metaData),1) )
metaData[2,2,2,2]=UInt32(1)
#setting offsets in metadata

twoTwoTwoOffset=0


metaBlock = 0
for xMetaa in 1:3,yMetaa in 1:3, zMetaa in 1:3 
  metaBlock+=20
  if(xMetaa==2 && yMetaa==2 && zMetaa==2 )
      twoTwoTwoOffset=metaBlock

  end

  for i in 1:14
    metaData[xMetaa,yMetaa,zMetaa,getResOffsetsBeg()+i]=i*1000+metaBlock*15*1000
    metaData[(xMetaa),(yMetaa),(zMetaa),(getIsToBeAnalyzedNumb() +i)] =1
  end
end


for xMetaa in 1:5,yMetaa in 1:5, zMetaa in 1:5 
  for i in 1:14
    metaData[(xMetaa),(yMetaa),(zMetaa),(getIsToBeAnalyzedNumb() +i)] =1
  end
end






isGold = 1
iterNumb = 1

resList = allocateResultLists(1000,1000)
inBlockLoopXZIterWithPadding = cld(32,dataBdim[2])
numberToLooFor = 2
resList = allocateResultLists(100000,100000)
workQueaue[:,1] = [2,2,2,1] 
workQueaue[:,2] = [3,3,3,1] 

workQueaue[:,3] = [4,3,3,1] 
workQueaue[:,4] = [2,3,3,1] 
workQueaue[:,5] = [3,4,3,1] 
workQueaue[:,6] = [3,2,3,1] 
workQueaue[:,7] = [3,3,4,1] 
workQueaue[:,8] = [3,3,2,1] 

workQueaue[:,9] = [1,2,2,1] 
workQueaue[:,10] = [3,2,2,1] 
workQueaue[:,11] = [2,1,2,1] 
workQueaue[:,12] = [2,3,2,1] 
workQueaue[:,13] = [2,2,1,1] 
workQueaue[:,14] = [2,2,3,1] 


workQueaue[:,15] = [5,5,1,1] 
workQueaue[:,16] = [5,5,2,1] 
workQueaue[:,17] = [5,5,3,1] 
workQueaue[:,18] = [5,5,4,1] 
workQueaue[:,19] = [5,5,5,1] 
workQueaue[:,20] = [5,5,6,1] 



workQueaueCounter[1] = 20
dilatationArrs= (mainArr,mainArr)
referenceArrs=(refArr,refArr)
shmemSumLengthMaxDiv4 = 20
paddingStore = CUDA.zeros(UInt8, metaDataDims[1],metaDataDims[2],metaDataDims[3],32,32)
function testProcessDataBlock(numberToLooFor,refArr,inBlockLoopXZIterWithPadding,paddingStore,shmemSumLengthMaxDiv4,referenceArrs,dilatationArrs,resList, metaData,metaDataDims,mainArrDims,isGold,iterNumb,mainArr,dataBdim,workQueaue,workQueaueCounter)
  MainLoopKernel.@mainLoopKernelAllocations(dataBdim)
  @ifXY 1 1 for i in 1:14
    areToBeValidated[i]=true
  end  
  @ifXY 3 1 goldToBeDilatated[1]=true
  @ifXY 3 1 segmToBeDilatated[1]=true

  sync_threads()



  #we get dilatation from block its padding will be analyzed later
  @iterateOverWorkQueue(workQueaueCounter,workQueaue
  ,shmemSumLengthMaxDiv4,begin 
  ProcessMainDataVerB.@executeDataIter(mainArrDims 
      ,dilatationArrs[shmemSum[shmemIndex*4+4]+1]
      ,referenceArrs[shmemSum[shmemIndex*4+4]+1]
      ,shmemSum[shmemIndex*4+1]#xMeta
      ,shmemSum[shmemIndex*4+2]#yMeta
      ,shmemSum[shmemIndex*4+3]#zMeta
      ,shmemSum[shmemIndex*4+4]#isGold
      ,iterationNumberShmem[1]#iterNumb
  )

  end ) 

  sync_grid(grid_handle)

 
  @iterateOverWorkQueue(workQueaueCounter,workQueaue
  ,shmemSumLengthMaxDiv4,begin  
  ProcessMainDataVerB.@executeIterPadding(dilatationArrs[shmemSum[shmemIndex*4+4]+1]
      ,referenceArrs[shmemSum[shmemIndex*4+4]+1]
      ,(shmemSum[shmemIndex*4+1])#xMeta
      ,(shmemSum[shmemIndex*4+2])#yMeta
      ,(shmemSum[shmemIndex*4+3])#zMeta
      ,shmemSum[shmemIndex*4+4]#isGold
      ,iterationNumberShmem[1]#iterNumb
      )

  end ) 


    return
end

@cuda threads=threads blocks=1 cooperative = true shmem = get_shmemMainKernel(dataBdim) testProcessDataBlock(numberToLooFor,refArr,inBlockLoopXZIterWithPadding,paddingStore,shmemSumLengthMaxDiv4,referenceArrs,dilatationArrs,resList, metaData,metaDataDims,mainArrDims,isGold,iterNumb,mainArr,dataBdim,workQueaue,workQueaueCounter)

#we need to test couple thing
#1) does dilateted data correctly was written to correct spot in the mainArr 
mainArr[1]
referenceArrs[1][1,1,1]
referenceArrs[2][1,1,1]

afterDil = UInt32(0)
for bitPos in [1,2,4,5,6,31,32]
    @setBitTo(afterDil,bitPos,true)
end
afterDil
nn = mainArr[33,dataBdim[2]+1,2]
isBit1AtPos(nn,32)
mainArr[32,dataBdim[2]+1,2]


#top 5,5,1
nn = paddingStore[5,5,1,2,2]
@test isBit1AtPos(nn,7)
#bottom 5,5,2
nn = paddingStore[5,5,2  ,2,2]
@test isBit1AtPos(nn,2)
#left 5,5,3
nn = paddingStore[5,5,3  ,5,2]
@test isBit1AtPos(nn,3)
#right 5,5,4
nn = paddingStore[5,5,4  ,5,2]
@test isBit1AtPos(nn,4)
#anterior 5,5,5
nn = paddingStore[5,5,5  ,2,5]
@test isBit1AtPos(nn,5)
#posterior 5,5,6
nn = paddingStore[5,5,6  ,2,5]
@test isBit1AtPos(nn,6)






# sum(resList)

@test mainArr[33,dataBdim[2]+1,2]== afterDil
@test mainArr[33,dataBdim[2]*2,2]== afterDil
@test mainArr[64,dataBdim[2]+1,2]== afterDil

Int64(mainArr[33,dataBdim[2]+1,2])

rowOne = UInt32(0)
@setBitTo(rowOne,1,true)
@setBitTo(rowOne,5,true)
@setBitTo(rowOne,32,true)

# aa = mainArr[33,dataBdim[2]+1,2]
aa= 126
for i in 1:32
  if(isBit1AtPos(aa,i))
    print("i $(i)  ")
  end  
end

isBit1AtPos(2147483665,dataBdim[3])

# sum(paddingStore)
# if(xm ==2 && ym==2 && zm==2 && threadIdxX()==1 && threadIdxY()==1)     CUDA.@cuprint "7  locArr $(locArr) sourceShmem  $(shmemblockData[threadIdxX(),threadIdxY(),1]) res $(shmemblockData[threadIdxX(),threadIdxY(),2]) \n"     end


  @test mainArr[33+1,dataBdim[2]+1,2]== rowOne
  @test mainArr[33+1,dataBdim[2]*2,2]== rowOne
  @test mainArr[64+1,dataBdim[2]+1,2]== rowOne

  @test mainArr[33-1,dataBdim[2]+1,2]== rowOne
  @test mainArr[33-1,dataBdim[2]*2,2]== rowOne
  @test mainArr[64-1,dataBdim[2]+1,2]== rowOne

for i in [-1,1] 
  @test mainArr[33,dataBdim[2]+1+i,2]== rowOne
  @test mainArr[33,dataBdim[2]*2+i,2]== rowOne
  @test mainArr[64,dataBdim[2]+1+i,2]== rowOne
end
fullOnes = 0
for bitPos in 1:32
    @setBitTo(fullOnesTobecome,bitPos,true)
end

#this should be full after dilatation
for xx in ((3*32)+1):((4*32)), yy in ((3*dataBdim[2])+1):((4*dataBdim[2]))
  @test mainArr[xx,yy,3]== fullOnes
end


uu = Array(paddingStore)[2,2,2,:,:]
uu[1,1]
Int64(sum(uu))


#3) weather the information is block full has been properly set in the metadata the same is to be activated ...
 @test metaData[2,2,2,getFullInGoldNumb()]==0
 @test metaData[3,3,3,getFullInGoldNumb()]==1

 @test metaData[2,2,2,getActiveGoldNumb()]==1

 @test metaData[2,2,2,getIsToBeActivatedInGoldNumb()]==1

for i in [-1,1] 
  @test metaData[2+i,2,2,getIsToBeActivatedInGoldNumb()]==1
  @test metaData[2+i,2,2,getIsToBeActivatedInGoldNumb()]==1
  @test metaData[2+i,2,2,getIsToBeActivatedInGoldNumb()]==1
end

for i in [-1,1] 
  @test metaData[2,2+i,2,getIsToBeActivatedInGoldNumb()]==1
  @test metaData[2,2+i,2,getIsToBeActivatedInGoldNumb()]==1
  @test metaData[2,2+i,2,getIsToBeActivatedInGoldNumb()]==1
end


for i in [-1,1] 
  @test metaData[2,2,2+i,getIsToBeActivatedInGoldNumb()]==1
  @test metaData[2,2,2+i,getIsToBeActivatedInGoldNumb()]==1
  @test metaData[2,2,2+i,getIsToBeActivatedInGoldNumb()]==1
end


#4)wheather in results  we have entries that should be present there so correct x,y,z and dir 
function checkIsInResList(resList,x,y,z,dir)::Bool
  for i in 1:length(resList)
      if(resList[i,:]== [x,y,z,isGold,dir,iterNumb])
        return true
      end
  end  
  return false
end  



shouldBeInResultSet = [
  [33,dataBdim[2]*2,34,isGold,5,iterNumb]  #from bottom
  ,[64,dataBdim[2]+1,63,isGold,6,iterNumb]  #from top
  ,[64-1,dataBdim[2]+1,64,isGold,1,iterNumb]  #from right
  ,[64+1,dataBdim[2]+1,64,isGold,2,iterNumb]  #from left
  ,[64,dataBdim[2]+2,64,isGold,4,iterNumb]  #from posterior
  ,[64,dataBdim[2],64,isGold,3,iterNumb]  #from anterior
  ]

  for entry in shouldBeInResultSet
    @test checkIsInResList(resList,entry[1],entry[2],entry[3],entry[5])
  end


for xx in ((3*32)+1):((4*32)), yy in ((3*dataBdim[2])+1):((4*dataBdim[2])), zz in ((3*32)+1):((4*32))
  if(iseven(zz))
    #top is 6 and is checked before bottom
    push!(shouldBeInResultSet,[xx,yy,zz,isGold,6,iterNumb] )
  end  
end


for entry in shouldBeInResultSet
  @test checkIsInResList(resList,entry[1],entry[2],entry[3],entry[5])
end


using CUDA

mean(CUDA.ones(8))

#5 check weather result counters are set to correct numbers


# #6) are the results in correct spots - weahter they are related to the ques the should be ...
# #where offset will be 0 for block 2,2,2 and 20000 for 3,3,3

  
# metaBlock = 0
# for xMetaa in 1:3,yMetaa in 1:3, zMetaa in 1:3 
#   metaBlock+=20
#   for i in 1:14
#     metaData[2,2,2,getResOffsetsBeg()+metaBlock]=i*1000
#   end
# end


# function getBegEnd(dir)::Bool
#   for i in 1:14
#     if(i == )
#     beg = i*1000+twoTwoTwoOffset*15*1000
#     endd = (i+1)*1000+twoTwoTwoOffset*15*1000
#   end
  
# end

# function checkIsInResListAndSpot(resList,x,y,z,dir)::Bool
#   for i in 1:length(resList)
#       if(resList[i,:]== [x,y,z,isGold,dir,iterNumb])
#         return true
#       end
#   end  
#   return false
# end  

# for entry in shouldBeInResultSet
#   @test checkIsInResList(resList,entry[1],entry[2],entry[3],entry[5])
# end
















