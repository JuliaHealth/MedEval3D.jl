using Revise, Parameters, Logging, Test
using CUDA
includet("./test/includeAllUseFullForTest.jl")
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
using ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils

##### iter data block
threads=(32,5);
blocks =17;
mainArrDims= (67,177,90);
dataBdim = (43,21,17);
mainArrCPU= falses(mainArrDims);
mainArrCPU[5,5,5]= true;
mainArrGPU = CuArray(mainArrCPU);
metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);
metaDataDims= size(metaData);
inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(dataBdim[1] ,threads[1]),fld(dataBdim[2] ,threads[2]),dataBdim[3]    );
#we are iterating here block by block sequentially
loopXMeta,loopYZMeta= (1,1)#(metaDataDims[1],fld(metaDataDims[2]*metaDataDims[3] ,blocks)  )

resShmem = CUDA.zeros(Bool,(dataBdim[1]+2,dataBdim[2]+2,dataBdim[3]+2 ));
sourceShmem = CUDA.zeros(Bool,(dataBdim));

function processDataKernel(mainArrGPU,sourceShmem,resShmem,mainArrDims,metaDataDims,loopXMeta,loopYZMeta,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ )
    @iter3dOuter(metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,
    begin 
        # @ifXY 1 1    CUDA.@cuprint "  xMeta $(xMeta) yMeta $(yMeta)  zMeta $(zMeta) \n"   

        @iterDataBlock(mainArrDims,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,xMeta,yMeta,zMeta,
        begin
            #@ifXY 1 1    CUDA.@cuprint "  x $(x) y $(y)  z $(z) \n"   

            maskBool=mainArrGPU[x,y,z]
            ProcessMainDataVerB.@processMaskData( maskBool)

    end)end)
    
    return
end


@cuda threads=threads blocks=blocks processDataKernel(mainArrGPU,sourceShmem,resShmem,mainArrDims,metaDataDims,loopXMeta,loopYZMeta,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ )
@test  resShmem[5+1,5+1,5+1]==false
@test  resShmem[7,6,6]==true
@test  resShmem[4+1,5+1,5+1]==true
@test  resShmem[5+1,4+1,5+1]==true
@test  resShmem[5+1,6+1,5+1]==true
@test  resShmem[5+1,5+1,4+1]==true
@test  resShmem[5+1,5+1,6+1]==true

########## loadMainValues
using Revise, Parameters, Logging, Test
using CUDA
includet("./test/includeAllUseFullForTest.jl")
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
using ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils

threads=(32,5);
blocks =17;
mainArrDims= (67,177,90);
dataBdim = (43,21,17);
mainArrCPU= falses(mainArrDims);
mainArrCPU[5,5,5]= true;
mainArrCPU[33,10,7]= true;
mainArrGPU = CuArray(mainArrCPU);
metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);
metaDataDims= size(metaData);
inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(dataBdim[1] ,threads[1]),fld(dataBdim[2] ,threads[2]),dataBdim[3]    );
#we are iterating here block by block sequentially
loopXMeta,loopYZMeta= (1,1)#(metaDataDims[1],fld(metaDataDims[2]*metaDataDims[3] ,blocks)  )

resShmem = CUDA.zeros(Bool,(dataBdim[1]+2,dataBdim[2]+2,dataBdim[3]+2 ));
sourceShmem = CUDA.zeros(Bool,(dataBdim));

function processDataKernel(mainArrGPU,sourceShmem,resShmem,mainArrDims,metaDataDims,loopXMeta,loopYZMeta,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ )
    @iter3dOuter(metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,
    begin 
        #@ifXY 1 1    CUDA.@cuprint "  xMeta $(xMeta) yMeta $(yMeta)  zMeta $(zMeta) \n"   

        @iterDataBlock(mainArrDims,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,xMeta,yMeta,zMeta,
        begin
            ProcessMainDataVerB.@loadMainValues( mainArrGPU,dataBdim)

    end)end)
    
    return
end

@cuda threads=threads blocks=blocks processDataKernel(mainArrGPU,sourceShmem,resShmem,mainArrDims,metaDataDims,loopXMeta,loopYZMeta,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ )
@test  resShmem[5+1,5+1,5+1]==false
@test  resShmem[7,6,6]==true
@test  resShmem[4+1,5+1,5+1]==true
@test  resShmem[5+1,4+1,5+1]==true
@test  resShmem[5+1,6+1,5+1]==true
@test  resShmem[5+1,5+1,4+1]==true
@test  resShmem[5+1,5+1,6+1]==true

@test sum(resShmem)==12

@test  resShmem[34,10,8]==true
@test  sourceShmem[5 ,5 ,5 ]==true
@test  sourceShmem[33 ,10 ,7 ]==true

filter(cart-> resShmem[cart], CartesianIndices(resShmem)  )

#################  paddingIter
using Revise, Parameters, Logging, Test
using CUDA
includet("./test/includeAllUseFullForTest.jl")
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
using ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils,..ResultListUtils

threads=(32,5);
blocks =1;
mainArrDims= (67,177,90);
 #dataBdim = (7,3,1);
dataBdim = (43,21,17);
mainArrCPU= falses(mainArrDims);
mainArrCPU[5,5,5]= true;
mainArrCPU[33,10,7]= true;
mainArrGPU = CuArray(mainArrCPU);
refArrCPU = falses(mainArrDims);
refArrGPU= CuArray(refArrCPU);

refArrCPU[dataBdim[1]+1,dataBdim[2]+2,dataBdim[3]+2]

metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);
metaDataDims= size(metaData);
totalFp,totalFn= 500,500
resList,resListIndicies,maxResListIndex= allocateResultLists(totalFp,totalFn)
iterNumb=3
isGold = 1
xMeta,yMeta,zMeta = 2,2,2
loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength =calculateLoopsIter(dataBdim,threads[1],threads[2],metaDataDims,blocks)
singleVal = CUDA.zeros(Int32,1)

#for start we will get top padding
function paddingIterKernel(singleVal,metaDataDims,refArrGPU,mainArrDims,loopAZFixed,loopBZfixed,dataBdim,mainArrGPU,iterNumb,isGold,xMeta,yMeta,zMeta)
    # resShmem =  @cuDynamicSharedMem(Bool,(dataBdim[1]+2,dataBdim[2]+2,dataBdim[3]+2)) # we need this additional 33th an 34th spots
    # sync_threads()
    # @ifXY 1 1 resShmem[1,2,2]= true
    # @ifXY 1 2 resShmem[1,2,3]= true
    # @ifXY 1 3 resShmem[1,3,2]= true

    # sync_threads()
    

    # paddingIter(loopX,loopY,maxXdim, maxYdim,a,b,c , xMetaChange,yMetaChange,zMetaChange, mainArr,refArr, dir,iterNumb,queueNumber,xMeta,yMeta,zMeta)

    @paddingIter(loopAZFixed,loopBZfixed,dataBdim[1], dataBdim[2],begin 
    CUDA.@atomic singleVal[1]+=1
end )


    return
end

@cuda threads=threads blocks=blocks paddingIterKernel(singleVal,metaDataDims,refArrGPU,mainArrDims,loopAZFixed,loopBZfixed,dataBdim,mainArrGPU,iterNumb,isGold,xMeta,yMeta,zMeta)
@test  sum(singleVal) == dataBdim[1]*dataBdim[2]

#################  paddingProcessCombined
using Revise, Parameters, Logging, Test
using CUDA
includet("./test/includeAllUseFullForTest.jl")
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
using ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils,..ResultListUtils

threads=(32,5);
blocks =1;
mainArrDims= (67,177,90);
dataBdim = (43,21,17);
mainArrCPU= falses(mainArrDims);
mainArrCPU[5,5,5]= true;
mainArrCPU[33,10,7]= true;
mainArrGPU = CuArray(mainArrCPU);
refArrCPU = falses(mainArrDims);


metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);
metaDataDims= size(metaData);
totalFp,totalFn= 500,500
resList,resListIndicies,maxResListIndex= allocateResultLists(totalFp,totalFn)
iterNumb=3
isGold = 1
queueNumber = 11
xMeta,yMeta,zMeta = 2,2,2
isToBeValidated = true
loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength = calculateLoopsIter(dataBdim,threads[1],threads[2],metaDataDims,blocks)
resShmem = CUDA.zeros(Bool,(dataBdim[1]+2,dataBdim[2]+2,dataBdim[3]+2)) # we need this additional 33th an 34th spots
resShmem[2,2,1]=true
resShmem[3,2,1]=true
resShmem[4,3,1]=true #for start we will get top padding
refArrCPU[dataBdim[1]+2,dataBdim[2]+2,dataBdim[3]+1]= true
refArrGPU= CuArray(refArrCPU);

resShmem[queueNumber,dataBdim[2],1]=true

dataBdim[1]+2
dataBdim[2]+2
dataBdim[3]+1
metaData[xMeta,yMeta,zMeta-1,getResOffsetsBeg()+queueNumber]=7

function paddingIterKernel(resShmem,metaData,resList,resListIndicies,maxResListIndex,metaDataDims,refArrGPU,mainArrDims,loopAZFixed,loopBZfixed,dataBdim,mainArrGPU,iterNumb,isGold,xMeta,yMeta,zMeta)

    # paddingIter(loopX,loopY,maxXdim, maxYdim,a,b,c , xMetaChange,yMetaChange,zMetaChange, mainArr,refArr, dir,iterNumb,queueNumber,xMeta,yMeta,zMeta)

    @paddingProcessCombined(loopAZFixed,loopBZfixed,dataBdim[1], dataBdim[2], x,y,1 ,0,0,-1,  mainArrGPU,refArrGPU,5,iterNumb, 12-isGold,xMeta,yMeta,zMeta,isGold)
    
    return
end

@cuda threads=threads blocks=blocks paddingIterKernel(resShmem,metaData,resList,resListIndicies,maxResListIndex,metaDataDims,refArrGPU,mainArrDims,loopAZFixed,loopBZfixed,dataBdim,mainArrGPU,iterNumb,isGold,xMeta,yMeta,zMeta)
@test Int64(sum(resListIndicies))>0
@test  Int64.(Array(resList[8,:]))==[dataBdim[1]+2,dataBdim[2]+2,dataBdim[3]+1,1,5,3]

######### process padding
using Revise, Parameters, Logging, Test
using CUDA
includet("./test/includeAllUseFullForTest.jl")
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
using ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils,..ResultListUtils

threads=(32,5);
blocks =1;
mainArrDims= (67,177,90);
dataBdim = (10,10,10)
mainArrCPU= falses(mainArrDims);
mainArrCPU[5,5,5]= true;
mainArrCPU[33,10,7]= true;
refArrCPU = falses(mainArrDims);


metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);
metaDataDims= size(metaData);
totalFp,totalFn= 500,500
resList,resListIndicies,maxResListIndex= allocateResultLists(totalFp,totalFn)
iterNumb=3
isGold = 1
queueNumber = 11
xMeta,yMeta,zMeta = 1,1,1 # here by assumption that we have all meta indicies 0 based
isToBeValidated = true
loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength = calculateLoopsIter(dataBdim,threads[1],threads[2],metaDataDims,blocks)
resShmem = CUDA.zeros(Bool,(dataBdim[1]+2,dataBdim[2]+2,dataBdim[3]+2)) # we need this additional 33th an 34th spots
# resShmem[2,2,1]=true
# resShmem[3,2,1]=true
# resShmem[4,3,1]=true #for start we will get top padding
refArrCPU[dataBdim[1]+2,dataBdim[2]+2,dataBdim[3]+1]= true
#setting some trues in both 
begin 

    # so we should get all mins as 2 and all maxes as 5
    
    # 1)   Left FP  
    # 2)   Left FN  
    mainArrCPU[11,13,13]=true
    refArrCPU[11,13,15]=true


   # 3)   Right FP 
   # 4)   Right FN  
    mainArrCPU[20,13,13]=true
    refArrCPU[20,13,15]=true

    mainArrCPU[20,14,13]=true
    refArrCPU[20,14,15]=true


 # 5)   Posterior FP  

    # 6)   Posterior FN  
 
    mainArrCPU[13,11,14]=true
    refArrCPU[14,11,13]=true

    mainArrCPU[15,11,13]=true
    refArrCPU[13,11,15]=true

    mainArrCPU[16,11,13]=true
    refArrCPU[13,11,16]=true


    # 7)   Anterior FP  
    # 8)   Anterior FN  
    mainArrCPU[14,20,13]=true
    refArrCPU[13,20,14]=true

    mainArrCPU[15,20,13]=true
    refArrCPU[14,20,16]=true

    mainArrCPU[17,20,13]=true
    refArrCPU[15,20,17]=true
    
    mainArrCPU[18,20,17]=true
    refArrCPU[16,20,18]=true
   

    # 9)   Top FP  
    # 10)   Top FN  
    mainArrCPU[14,15,11]=true
    refArrCPU[14,16,11]=true

    mainArrCPU[17,15,11]=true
    refArrCPU[17,16,11]=true

    mainArrCPU[15,15,11]=true
    refArrCPU[15,16,11]=true
    
    mainArrCPU[16,15,11]=true
    refArrCPU[16,16,11]=true

    mainArrCPU[18,15,11]=true
    refArrCPU[18,16,11]=true


    # 11)   Bottom FP  
    # 12)   Bottom FN  

    mainArrCPU[14,15,20]=true
    refArrCPU[14,16,20]=true

    mainArrCPU[17,15,20]=true
    refArrCPU[17,16,20]=true

    mainArrCPU[15,15,20]=true
    refArrCPU[15,16,20]=true
    
    mainArrCPU[16,15,20]=true
    refArrCPU[16,16,20]=true

    mainArrCPU[18,15,20]=true
    refArrCPU[18,16,20]=true

    mainArrCPU[19,15,20]=true
    refArrCPU[19,16,20]=true


    # 13)   main block Fp  
    # 14)   main block Fn  
    
    mainArrCPU[14,15,14]=true
    refArrCPU[14,16,16]=true

    mainArrCPU[17,15,17]=true
    refArrCPU[17,16,13]=true

    mainArrCPU[15,15,13]=true
    refArrCPU[15,16,13]=true
    
    mainArrCPU[16,15,13]=true
    refArrCPU[16,16,13]=true

    mainArrCPU[18,15,13]=true
    refArrCPU[18,16,13]=true

    mainArrCPU[19,15,13]=true
    refArrCPU[19,16,13]=true

    mainArrCPU[19,12,13]=true
    refArrCPU[19,12,13]=true

    mainArrCPU[12,12,17]=true
    refArrCPU[12,12,16]=true



    mainArrCPU[22,15,13]=true
    refArrCPU[18,22,13]=true

    mainArrCPU[19,22,13]=true
    refArrCPU[22,16,13]=true

    mainArrCPU[19,22,13]=true
    refArrCPU[19,12,22]=true

    mainArrCPU[12,12,17]=true
    refArrCPU[12,12,22]=true    

    mainArrCPU[9,15,13]=true
    refArrCPU[18,9,13]=true

    mainArrCPU[19,22,9]=true
    refArrCPU[9,16,13]=true

    mainArrCPU[19,9,13]=true
    refArrCPU[19,12,9]=true

    mainArrCPU[9,12,17]=true
    refArrCPU[12,9,22]=true
end  

mainArrGPU = CuArray(mainArrCPU);
## IMPORTANT in that way we set all points to agree so all should be aded to results !
# refArrGPU= CuArray(mainArrCPU);
refArrGPU= CUDA.ones(Bool,mainArrDims);
#setting offsets for all result queues
for qn in 1:14
    # resShmem[qn,1,1]=true
    #checking is it marked as to be validates
    # resShmem[qn,dataBdim[2],1]=true
    for a in [-1,0,1], b in [-1,0,1], c in [-1,0,1]
        metaData[xMeta+a+1,yMeta+b+1,zMeta+c+1,getResOffsetsBeg()+qn]=qn*50
    end    
end

inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(dataBdim[1] ,threads[1]),fld(dataBdim[2] ,threads[2]),dataBdim[3]    );
sourceShmem = CUDA.zeros(Bool,(dataBdim));
areToBeValidated= CUDA.ones(Bool,14)
isAnythingInPadding= CUDA.zeros(Bool,6)
function processPaddingKernel(isAnythingInPadding,areToBeValidated,sourceShmem,inBlockLoopX,inBlockLoopY,inBlockLoopZ,resShmem,metaData,resList,resListIndicies,maxResListIndex,metaDataDims,refArrGPU,mainArrDims,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,dataBdim,mainArrGPU,iterNumb,isGold,xMeta,yMeta,zMeta,isToBeValidated)

    # paddingIter(loopX,loopY,maxXdim, maxYdim,a,b,c , xMetaChange,yMetaChange,zMetaChange, mainArr,refArr, dir,iterNumb,queueNumber,xMeta,yMeta,zMeta)
    @iterDataBlock(mainArrDims,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,xMeta,yMeta,zMeta,
    begin
        ProcessMainDataVerB.@loadMainValues(mainArrGPU,dataBdim)
    end)

    sync_threads()

    @processPadding(isGold,xMeta,yMeta,zMeta,iterNumb,mainArrGPU,refArrGPU)

    return
end

@cuda threads=threads blocks=blocks processPaddingKernel(isAnythingInPadding,areToBeValidated,sourceShmem,inBlockLoopX,inBlockLoopY,inBlockLoopZ,resShmem,metaData,resList,resListIndicies,maxResListIndex,metaDataDims,refArrGPU,mainArrDims,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,dataBdim,mainArrGPU,iterNumb,isGold,xMeta,yMeta,zMeta,isToBeValidated)

@test Int64(metaData[1,2,2,getIsToBeActivatedInGoldNumb()])==1
@test Int64(metaData[3,2,2,getIsToBeActivatedInGoldNumb()])==1
@test Int64(metaData[2,1,2,getIsToBeActivatedInGoldNumb()])==1
@test Int64(metaData[2,3,2,getIsToBeActivatedInGoldNumb()])==1
@test Int64(metaData[2,2,1,getIsToBeActivatedInGoldNumb()])==1
@test Int64(metaData[2,2,3,getIsToBeActivatedInGoldNumb()])==1

metaCorrX = 2
metaCorrY = 2
metaCorrZ = 2
### gold pass analysis
# 1)   Left FP  so this should be only to the right 
@test Int64(metaData[metaCorrX+1,metaCorrY,metaCorrZ,getNewCountersBeg()+1])==2

# 3)   Right FP  
@test Int64(metaData[metaCorrX-1,metaCorrY,metaCorrZ,getNewCountersBeg()+3])==1

# 5)   Posterior FP  
@test Int64(metaData[metaCorrX,metaCorrY+1,metaCorrZ,getNewCountersBeg()+5])==4

# 7)   Anterior FP  
@test Int64(metaData[metaCorrX,metaCorrY-1,metaCorrZ,getNewCountersBeg()+7])==3

# 9)   Top FP  
@test Int64(metaData[metaCorrX,metaCorrY,metaCorrZ+1,getNewCountersBeg()+9])==6

# 11)   Bottom FP  
@test Int64(metaData[metaCorrX,metaCorrY,metaCorrZ-1,getNewCountersBeg()+11])==5


# 13)   main block Fp  
# @test Int64(metaData[metaCorrX,metaCorrY,metaCorrZ-1,getNewCountersBeg()+13])==8
@test length(filter(it->it>0,resListIndicies)) == (1+2+3+4+5+6)

@test sum(isAnythingInPadding)==6












######### process main part of data block
### data 1 
using Revise, Parameters, Logging, Test
using CUDA
includet("./test/includeAllUseFullForTest.jl")
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
using ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils,..ResultListUtils

threads=(32,5);
blocks =1;
mainArrDims= (67,177,90);
dataBdim = (10,10,10)
mainArrCPU= falses(mainArrDims);
mainArrCPU[5,5,5]= true;
mainArrCPU[33,10,7]= true;
refArrCPU = falses(mainArrDims);


metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);
metaDataDims= size(metaData);
totalFp,totalFn= 500,500
resList,resListIndicies,maxResListIndex= allocateResultLists(totalFp,totalFn)
iterNumb=3
isGold = 1
queueNumber = 11
xMeta,yMeta,zMeta = 1,1,1 # here by assumption that we have all meta indicies 0 based
isToBeValidated = true
loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength = calculateLoopsIter(dataBdim,threads[1],threads[2],metaDataDims,blocks)
resShmem = CUDA.zeros(Bool,(dataBdim[1]+2,dataBdim[2]+2,dataBdim[3]+2)) # we need this additional 33th an 34th spots
# resShmem[2,2,1]=true
# resShmem[3,2,1]=true
# resShmem[4,3,1]=true #for start we will get top padding
refArrCPU[dataBdim[1]+2,dataBdim[2]+2,dataBdim[3]+1]= true
#setting some trues in both 
begin 

    # so we should get all mins as 2 and all maxes as 5
    
    # 1)   Left FP  
    # 2)   Left FN  
    mainArrCPU[11,13,13]=true
    refArrCPU[11,13,15]=true


   # 3)   Right FP 
   # 4)   Right FN  
    mainArrCPU[20,13,13]=true
    refArrCPU[20,13,15]=true

    mainArrCPU[20,14,13]=true
    refArrCPU[20,14,15]=true


 # 5)   Posterior FP  

    # 6)   Posterior FN  
 
    mainArrCPU[13,11,14]=true
    refArrCPU[14,11,13]=true

    mainArrCPU[15,11,13]=true
    refArrCPU[13,11,15]=true

    mainArrCPU[16,11,13]=true
    refArrCPU[13,11,16]=true


    # 7)   Anterior FP  
    # 8)   Anterior FN  
    mainArrCPU[14,20,13]=true
    refArrCPU[13,20,14]=true

    mainArrCPU[15,20,13]=true
    refArrCPU[14,20,16]=true

    mainArrCPU[17,20,13]=true
    refArrCPU[15,20,17]=true
    
    mainArrCPU[18,20,17]=true
    refArrCPU[16,20,18]=true
   

    # 9)   Top FP  
    # 10)   Top FN  
    mainArrCPU[14,15,11]=true
    refArrCPU[14,16,11]=true

    mainArrCPU[17,15,11]=true
    refArrCPU[17,16,11]=true

    mainArrCPU[15,15,11]=true
    refArrCPU[15,16,11]=true
    
    mainArrCPU[16,15,11]=true
    refArrCPU[16,16,11]=true

    mainArrCPU[18,15,11]=true
    refArrCPU[18,16,11]=true


    # 11)   Bottom FP  
    # 12)   Bottom FN  

    mainArrCPU[14,15,20]=true
    refArrCPU[14,16,20]=true

    mainArrCPU[17,15,20]=true
    refArrCPU[17,16,20]=true

    mainArrCPU[15,15,20]=true
    refArrCPU[15,16,20]=true
    
    mainArrCPU[16,15,20]=true
    refArrCPU[16,16,20]=true

    mainArrCPU[18,15,20]=true
    refArrCPU[18,16,20]=true

    mainArrCPU[19,15,20]=true
    refArrCPU[19,16,20]=true


    # 13)   main block Fp  
    # 14)   main block Fn  
    
    mainArrCPU[14,15,14]=true
    refArrCPU[14,16,16]=true

    mainArrCPU[17,15,17]=true
    refArrCPU[17,16,13]=true

    mainArrCPU[15,15,13]=true
    refArrCPU[15,16,13]=true
    
    mainArrCPU[16,15,13]=true
    refArrCPU[16,16,13]=true

    mainArrCPU[18,15,13]=true
    refArrCPU[18,16,13]=true

    mainArrCPU[19,15,13]=true
    refArrCPU[19,16,13]=true

    mainArrCPU[19,12,13]=true
    refArrCPU[19,12,13]=true

    mainArrCPU[12,12,17]=true
    refArrCPU[12,12,16]=true



    mainArrCPU[22,15,13]=true
    refArrCPU[18,22,13]=true

    mainArrCPU[19,22,13]=true
    refArrCPU[22,16,13]=true

    mainArrCPU[19,22,13]=true
    refArrCPU[19,12,22]=true

    mainArrCPU[12,12,17]=true
    refArrCPU[12,12,22]=true    

    mainArrCPU[9,15,13]=true
    refArrCPU[18,9,13]=true

    mainArrCPU[19,22,9]=true
    refArrCPU[9,16,13]=true

    mainArrCPU[19,9,13]=true
    refArrCPU[19,12,9]=true

    mainArrCPU[9,12,17]=true
    refArrCPU[12,9,22]=true
end  

mainArrGPU = CuArray(mainArrCPU);
## IMPORTANT in that way we set all points to agree so all should be aded to results !
# refArrGPU= CuArray(mainArrCPU);
refArrGPU= CUDA.ones(Bool,mainArrDims);
#setting offsets for all result queues
for qn in 1:14
    # resShmem[qn,1,1]=true
    #checking is it marked as to be validates
    # resShmem[qn,dataBdim[2],1]=true
    for a in [-1,0,1], b in [-1,0,1], c in [-1,0,1]
        metaData[xMeta+a+1,yMeta+b+1,zMeta+c+1,getResOffsetsBeg()+qn]=qn*50
        metaData[xMeta+a+1,yMeta+b+1,zMeta+c+1,getIsToBeAnalyzedNumb()+qn]=1
    end    
end

inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(dataBdim[1] ,threads[1]),fld(dataBdim[2] ,threads[2]),dataBdim[3]    );
sourceShmem = CUDA.zeros(Bool,(dataBdim));
areToBeValidated= CUDA.ones(Bool,14)
isAnythingInPadding= CUDA.zeros(Bool,6)
function executeDataIterWithPaddingKernel(isAnythingInPadding,areToBeValidated,sourceShmem,inBlockLoopX,inBlockLoopY,inBlockLoopZ,resShmem,metaData,resList,resListIndicies,maxResListIndex,metaDataDims,refArrGPU,mainArrDims,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,dataBdim,mainArrGPU,iterNumb,isGold,xMeta,yMeta,zMeta,isToBeValidated)
    isMaskFull= true
    @executeDataIterWithPadding(mainArrDims, inBlockLoopX,inBlockLoopY,inBlockLoopZ,mainArrGPU,refArrGPU,xMeta,yMeta,zMeta,isGold,iterNumb)

    return
end

@cuda threads=threads blocks=blocks executeDataIterWithPaddingKernel(isAnythingInPadding,areToBeValidated,sourceShmem,inBlockLoopX,inBlockLoopY,inBlockLoopZ,resShmem,metaData,resList,resListIndicies,maxResListIndex,metaDataDims,refArrGPU,mainArrDims,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,dataBdim,mainArrGPU,iterNumb,isGold,xMeta,yMeta,zMeta,isToBeValidated)

@test Int64(metaData[1,2,2,getIsToBeActivatedInGoldNumb()])==1
@test Int64(metaData[3,2,2,getIsToBeActivatedInGoldNumb()])==1
@test Int64(metaData[2,1,2,getIsToBeActivatedInGoldNumb()])==1
@test Int64(metaData[2,3,2,getIsToBeActivatedInGoldNumb()])==1
@test Int64(metaData[2,2,1,getIsToBeActivatedInGoldNumb()])==1
@test Int64(metaData[2,2,3,getIsToBeActivatedInGoldNumb()])==1

metaCorrX = 2
metaCorrY = 2
metaCorrZ = 2
### gold pass analysis
# 1)   Left FP  so this should be only to the right 
@test Int64(metaData[metaCorrX+1,metaCorrY,metaCorrZ,getNewCountersBeg()+1])==2

# 3)   Right FP  
@test Int64(metaData[metaCorrX-1,metaCorrY,metaCorrZ,getNewCountersBeg()+3])==1

# 5)   Posterior FP  
@test Int64(metaData[metaCorrX,metaCorrY+1,metaCorrZ,getNewCountersBeg()+5])==4

# 7)   Anterior FP  
@test Int64(metaData[metaCorrX,metaCorrY-1,metaCorrZ,getNewCountersBeg()+7])==3

# 9)   Top FP  
@test Int64(metaData[metaCorrX,metaCorrY,metaCorrZ+1,getNewCountersBeg()+9])==6

# 11)   Bottom FP  
@test Int64(metaData[metaCorrX,metaCorrY,metaCorrZ-1,getNewCountersBeg()+11])==5


@test  metaData[metaCorrX,metaCorrY,metaCorrZ,getFullInSegmNumb()-1]==0

########## data 2

using Revise, Parameters, Logging, Test
using CUDA
includet("./test/includeAllUseFullForTest.jl")
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
using ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils,..ResultListUtils

threads=(32,5);
blocks =1;
mainArrDims= (67,177,90);
dataBdim = (10,10,10)
mainArrCPU= falses(mainArrDims);

mainArrCPU[17,17,17]= true;
refArrCPU = falses(mainArrDims);

metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);
metaDataDims= size(metaData);
totalFp,totalFn= 500,500
resList,resListIndicies,maxResListIndex= allocateResultLists(totalFp,totalFn)
iterNumb=3
isGold = 1
queueNumber = 11
xMeta,yMeta,zMeta = 1,1,1 # here by assumption that we have all meta indicies 0 based
isToBeValidated = true
loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength = calculateLoopsIter(dataBdim,threads[1],threads[2],metaDataDims,blocks)
resShmem = CUDA.zeros(Bool,(dataBdim[1]+2,dataBdim[2]+2,dataBdim[3]+2)) # we need this additional 33th an 34th spots

mainArrGPU = CuArray(mainArrCPU);
## IMPORTANT in that way we set all points to agree so all should be aded to results !
# refArrGPU= CuArray(mainArrCPU);
refArrGPU= CUDA.ones(Bool,mainArrDims);
#setting offsets for all result queues
for qn in 1:14
    # resShmem[qn,1,1]=true
    #checking is it marked as to be validates
    # resShmem[qn,dataBdim[2],1]=true
    for a in [-1,0,1], b in [-1,0,1], c in [-1,0,1]
        metaData[xMeta+a+1,yMeta+b+1,zMeta+c+1,getResOffsetsBeg()+qn]=qn*50
        metaData[xMeta+a+1,yMeta+b+1,zMeta+c+1,getIsToBeAnalyzedNumb()+qn]=1
    end    
end

inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(dataBdim[1] ,threads[1]),fld(dataBdim[2] ,threads[2]),dataBdim[3]    );
sourceShmem = CUDA.zeros(Bool,(dataBdim));
areToBeValidated= CUDA.ones(Bool,14)
isAnythingInPadding= CUDA.zeros(Bool,6)
function executeDataIterWithPaddingKernel(isAnythingInPadding,areToBeValidated,sourceShmem,inBlockLoopX,inBlockLoopY,inBlockLoopZ,resShmem,metaData,resList,resListIndicies,maxResListIndex,metaDataDims,refArrGPU,mainArrDims,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,dataBdim,mainArrGPU,iterNumb,isGold,xMeta,yMeta,zMeta,isToBeValidated)
    isMaskFull= true
    @executeDataIterWithPadding(mainArrDims, inBlockLoopX,inBlockLoopY,inBlockLoopZ,mainArrGPU,refArrGPU,xMeta,yMeta,zMeta,isGold,iterNumb)

    return
end

@cuda threads=threads blocks=blocks executeDataIterWithPaddingKernel(isAnythingInPadding,areToBeValidated,sourceShmem,inBlockLoopX,inBlockLoopY,inBlockLoopZ,resShmem,metaData,resList,resListIndicies,maxResListIndex,metaDataDims,refArrGPU,mainArrDims,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,dataBdim,mainArrGPU,iterNumb,isGold,xMeta,yMeta,zMeta,isToBeValidated)

metaCorrX = 2
metaCorrY = 2
metaCorrZ = 2
@test  metaData[metaCorrX,metaCorrY,metaCorrZ,getFullInSegmNumb()-1]==0

# 13)   main block Fp  
# @test Int64(metaData[metaCorrX,metaCorrY,metaCorrZ-1,getNewCountersBeg()+13])==8


### gold pass analysis
# 1)   Left FP  so this should be only to the right 
@test Int64(metaData[metaCorrX,metaCorrY,metaCorrZ,getNewCountersBeg()+13])==6

length(filter(it->it>0,resListIndicies))
########## data 3

using Revise, Parameters, Logging, Test
using CUDA
includet("./test/includeAllUseFullForTest.jl")
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
using ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils,..ResultListUtils

threads=(32,5);
blocks =1;
mainArrDims= (67,177,90);
dataBdim = (10,10,10)
mainArrCPU= trues(mainArrDims);

mainArrCPU[17,17,17]= true;
refArrCPU = falses(mainArrDims);

metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);
metaDataDims= size(metaData);
totalFp,totalFn= 500,500
resList,resListIndicies,maxResListIndex= allocateResultLists(totalFp,totalFn)
iterNumb=3
isGold = 1
queueNumber = 11
xMeta,yMeta,zMeta = 1,1,1 # here by assumption that we have all meta indicies 0 based
isToBeValidated = true
loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength = calculateLoopsIter(dataBdim,threads[1],threads[2],metaDataDims,blocks)
resShmem = CUDA.zeros(Bool,(dataBdim[1]+2,dataBdim[2]+2,dataBdim[3]+2)) # we need this additional 33th an 34th spots

mainArrGPU = CuArray(mainArrCPU);
## IMPORTANT in that way we set all points to agree so all should be aded to results !
# refArrGPU= CuArray(mainArrCPU);
refArrGPU= CUDA.ones(Bool,mainArrDims);
#setting offsets for all result queues
for qn in 1:14
    # resShmem[qn,1,1]=true
    #checking is it marked as to be validates
    # resShmem[qn,dataBdim[2],1]=true
    for a in [-1,0,1], b in [-1,0,1], c in [-1,0,1]
        metaData[xMeta+a+1,yMeta+b+1,zMeta+c+1,getResOffsetsBeg()+qn]=qn*50
        metaData[xMeta+a+1,yMeta+b+1,zMeta+c+1,getIsToBeAnalyzedNumb()+qn]=1
    end    
end

inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(dataBdim[1] ,threads[1]),fld(dataBdim[2] ,threads[2]),dataBdim[3]    );
sourceShmem = CUDA.zeros(Bool,(dataBdim));
areToBeValidated= CUDA.ones(Bool,14)
isAnythingInPadding= CUDA.zeros(Bool,6)
function executeDataIterWithPaddingKernel(isAnythingInPadding,areToBeValidated,sourceShmem,inBlockLoopX,inBlockLoopY,inBlockLoopZ,resShmem,metaData,resList,resListIndicies,maxResListIndex,metaDataDims,refArrGPU,mainArrDims,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,dataBdim,mainArrGPU,iterNumb,isGold,xMeta,yMeta,zMeta,isToBeValidated)
    isMaskFull= true
    @executeDataIterWithPadding(mainArrDims, inBlockLoopX,inBlockLoopY,inBlockLoopZ,mainArrGPU,refArrGPU,xMeta,yMeta,zMeta,isGold,iterNumb)

    return
end

@cuda threads=threads blocks=blocks executeDataIterWithPaddingKernel(isAnythingInPadding,areToBeValidated,sourceShmem,inBlockLoopX,inBlockLoopY,inBlockLoopZ,resShmem,metaData,resList,resListIndicies,maxResListIndex,metaDataDims,refArrGPU,mainArrDims,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,dataBdim,mainArrGPU,iterNumb,isGold,xMeta,yMeta,zMeta,isToBeValidated)

metaCorrX = 2
metaCorrY = 2
metaCorrZ = 2
@test  metaData[metaCorrX,metaCorrY,metaCorrZ,getFullInSegmNumb()-1]==1