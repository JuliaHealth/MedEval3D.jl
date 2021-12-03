
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




mainArr= CUDA.zeros(UInt32, 500,500,30)
refArr= CUDA.zeros(UInt8, 500,500,1000)
numbToLooFor = 2
dataBdim= (32,10,32)


############ some arbitrary data  meta 2,2,2 with some results to be set 
xMeta,yMeta,zMeta = 2,2,2

resShmemblockData= CUDA.zeros(UInt32, 32,10);
shmemblockData= CUDA.zeros(UInt32, 32,10);
shmemPaddings= CUDA.zeros(Bool, 32,32,6);

threads=(32,10)
rowOne = 0

#below we will organize it so we should have results in all neghbouring blocks - also from corners 
@setBitTo(rowOne,1,true)
@setBitTo(rowOne,5,true)
@setBitTo(rowOne,32,true)

mainArr[33,11,2]= rowOne
mainArr[33,20,2]= rowOne
mainArr[64,11,2]= rowOne
mainArr[64,20,2]= rowOne

mainArr[64,11,2]= rowOne


################   here we will get a block that will become full after the dilatation meta 3,3,3
xMeta,yMeta,zMeta = 4,4,4
fullOnesTobecome = 0
for bitPos in 1:32
  if(isodd(bitPos))
    @setBitTo(fullOnesTobecome,bitPos,true)
  end
end
for xx in ((3*32)+1):((4*32)), yy in ((3*10)+1):((4*10))
  mainArr[xx,yy,3]= fullOnesTobecome
end

for xx in ((3*32)+1):((4*32)), yy in ((3*10)+1):((4*10)), zz in ((3*32)+1):((4*32))
  refArr[xx,yy,zz]= 2
end

refArr[33,11,33]= 2
refArr[33,20,34]= 2
refArr[64,11,37]= 2
refArr[64,11,38]= 2
refArr[64,11,64]= 2


### configurations

blocks =1
mainArrDims= size(mainArr)
metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim)
#metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,dataBdim),1:9,2:3,4:6,: );
metaDataDims=size(metaData)

workQueaue,workQueaueCounter= WorkQueueUtils.allocateWorkQueue( max(length(metaData),1) )
metaData[2,2,2,2]=UInt32(1)
#setting offsets in metadata

metaBlock = 0
for xMetaa in 1:3,yMetaa in 1:3, zMetaa in 1:3 
  metaBlock+=20
  for i in 1:14
    metaData[2,2,2,getResOffsetsBeg()+i]=i*1000+metaBlock*15*1000
  end
end


isGold = 1
iterNumb = 1

resList = allocateResultLists(1000,1000)
inBlockLoopXZIterWithPadding = cld(32,10)
numberToLooFor = 2
resList = allocateResultLists(100000,100000)
workQueaue[:,1] = [1,1,1,1] 
workQueaue[:,2] = [3,3,3,1] 
workQueaueCounter[1] = 2
dilatationArrs= (mainArr,mainArr)
referenceArrs=(refArr,refArr)
shmemSumLengthMaxDiv4 = 5
paddingStore = CUDA.zeros(UInt8, metaDataDims[1],metaDataDims[2],metaDataDims[3],32,32)
function testProcessDataBlock(refArr,inBlockLoopXZIterWithPadding,paddingStore,shmemSumLengthMaxDiv4,referenceArrs,dilatationArrs,resList, metaData,metaDataDims,mainArrDims,isGold,iterNumb,mainArr,dataBdim,workQueaue,workQueaueCounter)
  MainLoopKernel.@mainLoopKernelAllocations(dataBdim)
  @ifXY 1 1 for i in 1:14
    areToBeValidated[i]=true
  end  
  
  sync_threads()


 
 #  @ifXY 1 1 "aaaaaaaaa  $(Int64(((dilatationArrs[1])[1] )))  \n"

  # @iterateOverWorkQueue(workQueaueCounter,workQueaue
  # ,shmemSumLengthMaxDiv4,begin     
  # ProcessMainDataVerB.@executeIterPadding(dilatationArrs[shmemSum[shmemIndex*4+4]+1]
  #     ,referenceArrs[shmemSum[shmemIndex*4+4]+1]
  #     ,shmemSum[shmemIndex*4+1]#xMeta
  #     ,shmemSum[shmemIndex*4+2]#yMeta
  #     ,shmemSum[shmemIndex*4+3]#zMeta
  #     ,shmemSum[shmemIndex*4+4]#isGold
  #     ,iterationNumberShmem[1]#iterNumb
  #     )

  # end ) 
 sync_grid(grid_handle)

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

    return
end

@cuda threads=threads blocks=blocks shmem = get_shmemMainKernel(dataBdim) testProcessDataBlock(refArr,inBlockLoopXZIterWithPadding,paddingStore,shmemSumLengthMaxDiv4,referenceArrs,dilatationArrs,resList, metaData,metaDataDims,mainArrDims,isGold,iterNumb,mainArr,dataBdim,workQueaue,workQueaueCounter)

#we need to test couple thing
#1) does dilateted data correctly was written to correct spot in the mainArr 
mainArr[1]
referenceArrs[1][1,1,1]
referenceArrs[2][1,1,1]

afterDil = 0
for bitPos in [1,2,4,5,6,31,32]
    @setBitTo(afterDil,bitPos,true)
end


@test mainArr[33,11,2]== afterDil
@test mainArr[33,20,2]== afterDil
@test mainArr[64,11,2]== afterDil

rowOne = 0
@setBitTo(rowOne,1,true)
@setBitTo(rowOne,5,true)
@setBitTo(rowOne,32,true)

for i in [-1,1] 
  @test mainArr[33+i,11,2]== rowOne
  @test mainArr[33+i,20,2]== rowOne
  @test mainArr[64+i,11,2]== rowOne
end

for i in [-1,1] 
  @test mainArr[33,11+i,2]== rowOne
  @test mainArr[33,20+i,2]== rowOne
  @test mainArr[64,11+i,2]== rowOne
end
fullOnes = 0
for bitPos in 1:32
    @setBitTo(fullOnesTobecome,bitPos,true)
end

#this should be full after dilatation
for xx in ((3*32)+1):((4*32)), yy in ((3*10)+1):((4*10))
  @test mainArr[xx,yy,3]== fullOnes
end


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

@test checkIsInResList(resList,x,y,z,dir)

# refArr[33,11,33]= 2
# refArr[33,20,34]= 2
# refArr[64,11,37]= 2
# refArr[64,11,38]= 2
# refArr[64,11,64]= 2
# for xx in ((3*32)+1):((4*32)), yy in ((3*10)+1):((4*10)), zz in ((3*32)+1):((4*32))
#   refArr[xx,yy,zz]= 2
# end
# @setBitTo(rowOne,1,true)
# @setBitTo(rowOne,5,true)
# @setBitTo(rowOne,32,true)

# mainArr[33,11,2]= rowOne
# mainArr[33,20,2]= rowOne
# mainArr[64,11,2]= rowOne
# mainArr[64,11,2]= rowOne
# mainArr[64,11,2]= rowOne









#5 check weather result counters are set to correct numbers


#6) are the results in correct spots - weahter they are related to the ques the should be ...
#where offset will be 0 for block 2,2,2 and 20000 for 3,3,3

  
metaBlock = 0
for xMetaa in 1:3,yMetaa in 1:3, zMetaa in 1:3 
  metaBlock+=20
  for i in 1:14
    metaData[2,2,2,getResOffsetsBeg()+metaBlock]=i*1000
  end
end

















