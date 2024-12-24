using Revise, Parameters, Logging, Test
using CUDA
includet("./test/includeAllUseFullForTest.jl")
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
using ..ResultListUtils, ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils, ..ScanForDuplicates

resList = allocateResultList(100,100);
ind= (1,2,3)
getResLinIndex([ind[1] ind[2] ind[3] 1 1 1 ; 0 0 0 0 0 0],1, (200,200,200)) 

carts = CartesianIndices(zeros(200,200,200))

mappedA = map(ind-> getResLinIndex(ind[1],ind[2],ind[3],1, (200,200,200)) ,carts)
mappedB = map(ind-> getResLinIndex(ind[1],ind[2],ind[3],0, (200,200,200)) ,carts)

@test length(unique(mappedA))== 200*200*200
@test length(unique(vec(vcat(mappedA,mappedB ))))== 200*200*200*2



#################  addResult
using Revise, Parameters, Logging, Test
using CUDA
includet("./test/includeAllUseFullForTest.jl")
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
using ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils,..ResultListUtils

threads=(32,5);
blocks =2;
mainArrDims= (67,177,90);
dataBdim = (43,21,17);
mainArrCPU= falses(mainArrDims);
mainArrCPU[5,5,5]= true;
mainArrCPU[33,10,7]= true;
mainArrGPU = CuArray(mainArrCPU);
metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);
metaDataDims= size(metaData);
totalFp,totalFn= 500,500
resList,resListIndicies,maxResListIndex= allocateResultLists(totalFp,totalFn)

xMeta,yMeta,zMeta= 1,1,1 #aaume 0 based
x,y,z = 1,2,3
dir= 5
queueNumber = 1
isGold=4
iterNumb=6
metaData[xMeta+1,yMeta+1,zMeta+1,getResOffsetsBeg()+queueNumber]=7
function addResKernel(metaData ,xMeta,yMeta,zMeta, resList,resListIndicies,maxResListIndex,x,y,z, dir,iterNumb,queueNumber,metaDataDims ,mainArrDims,isGold )
    @ifXY 1 1 @addResult(metaData ,xMeta,yMeta,zMeta, resList,resListIndicies,maxResListIndex,x,y,z, dir,iterNumb,queueNumber,metaDataDims,mainArrDims ,isGold  )     
    @ifXY 2 1 @addResult(metaData ,xMeta,yMeta,zMeta, resList,resListIndicies,maxResListIndex,x,y,z, dir,iterNumb,queueNumber,metaDataDims ,mainArrDims,isGold  )     
    @ifXY 3 1 @addResult(metaData ,xMeta,yMeta,zMeta, resList,resListIndicies,maxResListIndex,x,y,z, dir,iterNumb,queueNumber,metaDataDims,mainArrDims ,isGold  )     
    return
end

@cuda threads=threads blocks=blocks addResKernel(metaData ,xMeta,yMeta,zMeta, resList,resListIndicies,maxResListIndex,x,y,z, dir,iterNumb,queueNumber,metaDataDims ,mainArrDims,isGold )
@test length(filter(it-> it>0,Array(resListIndicies)))==6

# @test  Int64.(Array(resList[7,:]))==[1,2,3,4,5,6]
@test  Int64.(Array(resList[8,:]))==[1,2,3,4,5,6]
@test  Int64.(Array(resList[9,:]))==[1,2,3,4,5,6]
@test  Int64.(Array(resList[10,:]))==[1,2,3,4,5,6]
@test  Int64.(Array(resList[11,:]))==[1,2,3,4,5,6]
@test  Int64.(Array(resList[12,:]))==[1,2,3,4,5,6]
@test  Int64.(Array(resList[14,:]))==[0,0,0,0,0,0]



Int64.(Array(resListIndicies[5:15]))

filter(it-> it>0,Array(resListIndicies))
Int64(sum(resList))