
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
using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.BitWiseUtils,Main.ResultListUtils, Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils, Main.ScanForDuplicates




mainArr= CUDA.zeros(UInt32, 50,50,10)
refArr= CUDA.zeros(UInt32, 50,50,10)
targetArr= CUDA.zeros(UInt32, 50,50,10)

dataBdim= (32,10,32)

xMeta,yMeta,zMeta = 1,1,1

resShmemblockData= CUDA.zeros(UInt32, 32,10);
shmemblockData= CUDA.zeros(UInt32, 32,10);
shmemPaddings= CUDA.zeros(Bool, 32,32,6);

threads=(32,10)
rowOne = 0
@setBitTo(rowOne,1,true)
@setBitTo(rowOne,5,true)
@setBitTo(rowOne,32,true)

mainArr[1,1,1]= rowOne
mainArr[1,10,1]= rowOne
mainArr[32,1,1]= rowOne

rowB = 0
@setBitTo(rowB,1,true)
@setBitTo(rowB,2,true)
@setBitTo(rowB,5,true)
@setBitTo(rowB,6,true)
@setBitTo(rowB,32,true)

targetArr[1,10,1]= rowB




blocks =1
mainArrDims= (50,50,320)
metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim)
#metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,dataBdim),1:9,2:3,4:6,: );
metaDataDims=size(metaData)

workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter= WorkQueueUtils.allocateWorkQueue( max(length(metaData),1) )
metaData[2,2,2,2]=UInt32(1)
#setting offsets in metadata
for i in 1:14
  metaData[xMeta,yMeta,zMeta,getResOffsetsBeg()+i]=i*10
end
isGold = 1
iterNumb = 1
function testvalidateData(shmemPaddings,shmemblockData,resShmemblockData,metaData,metaDataDims,mainArrDims,isGold,xMeta,yMeta,zMeta,iterNumb,mainArr,refArr,targetArr,dataBdim,workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter)
  @loadMainValues(mainArr,xMeta,yMeta,zMeta)

  sync_threads()
  @validateData(isGold,xMeta,yMeta,zMeta,iterNumb,mainArr,refArr,targetArr)

    return
end

@cuda threads=threads blocks=blocks testvalidateData(shmemPaddings,shmemblockData,resShmemblockData,metaData,metaDataDims,mainArrDims,isGold,xMeta,yMeta,zMeta,iterNumb,mainArr,refArr,targetArr,dataBdim,workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter)

@test shmemPaddings[1,1,4]
@test shmemPaddings[5,1,4]
@test shmemPaddings[32,1,4]
@test !shmemPaddings[32,2,4]

@test !shmemPaddings[32,2,4]

@setBitTo(rowB,1,true)
@setBitTo(rowB,2,true)
@setBitTo(rowB,5,true)
@setBitTo(rowB,6,true)
@setBitTo(rowB,32,true)

targetArr[1,10,1]= rowB








################## execute Data Iter With Padding  first data 

using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.BitWiseUtils,Main.ResultListUtils, Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils, Main.ScanForDuplicates




mainArr= CUDA.zeros(UInt32, 50,50,10)
refArr= CUDA.zeros(UInt32, 50,50,10)
targetArr= CUDA.zeros(UInt32, 50,50,10)

dataBdim= (32,10,32)

xMeta,yMeta,zMeta = 2,2,2

resShmemblockData= CUDA.zeros(UInt32, 32,10);
shmemblockData= CUDA.zeros(UInt32, 32,10);
shmemPaddings= CUDA.zeros(Bool, 32,32,6);

threads=(32,10)
rowOne = 0
@setBitTo(rowOne,1,true)
@setBitTo(rowOne,5,true)
@setBitTo(rowOne,32,true)

mainArr[33,11,2]= rowOne
mainArr[33,20,2]= rowOne
mainArr[64,11,2]= rowOne

rowB = 0
@setBitTo(rowB,1,true)
@setBitTo(rowB,2,true)
@setBitTo(rowB,5,true)
@setBitTo(rowB,6,true)
@setBitTo(rowB,32,true)

targetArr[33,20,2]= rowB




blocks =1
mainArrDims= (50,50,320)
metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim)
#metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,dataBdim),1:9,2:3,4:6,: );
metaDataDims=size(metaData)

workQueue,workQueueCounter= WorkQueueUtils.allocateWorkQueue( max(length(metaData),1) )
metaData[2,2,2,2]=UInt32(1)
#setting offsets in metadata
for i in 1:14
  metaData[xMeta,yMeta,zMeta,getResOffsetsBeg()+i]=i*10
end
isGold = 1
iterNumb = 1

resList = allocateResultLists(1000,1000)

function testvalidateData(resList               shmemPaddings,shmemblockData,resShmemblockData,metaData,metaDataDims,mainArrDims,isGold,xMeta,yMeta,zMeta,iterNumb,mainArr,refArr,targetArr,dataBdim,workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter)
  @loadMainValues(mainArr,xMeta,yMeta,zMeta)

  sync_threads()
  @validateData(isGold,xMeta,yMeta,zMeta,iterNumb,mainArr,refArr,targetArr)

    return
end

@cuda threads=threads blocks=blocks testvalidateData(shmemPaddings,shmemblockData,resShmemblockData,metaData,metaDataDims,mainArrDims,isGold,xMeta,yMeta,zMeta,iterNumb,mainArr,refArr,targetArr,dataBdim,workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter)

@test shmemPaddings[1,1,4]
@test shmemPaddings[5,1,4]
@test shmemPaddings[32,1,4]
@test !shmemPaddings[32,2,4]

@test !shmemPaddings[32,2,4]

@setBitTo(rowB,1,true)
@setBitTo(rowB,2,true)
@setBitTo(rowB,5,true)
@setBitTo(rowB,6,true)
@setBitTo(rowB,32,true)

targetArr[1,10,1]= rowB









