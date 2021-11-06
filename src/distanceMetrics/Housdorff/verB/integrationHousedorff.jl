######### getting all together in Housedorff 

using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils,Main.ResultListUtils

threads=(32,5);
blocks =1;
mainArrDims= (67,177,90);
dataBdim = (10,10,10)
mainArrCPU= falses(mainArrDims);
refArrCPU = falses(mainArrDims);
##### we will create two planes 20 units apart from each 
mainArrCPU[10:50,10:50,10]= true
refArrCPU[10:50,10:50,30]= true
metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);
metaDataDims= size(metaData);
loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ= ProcessMainDataVerB.calculateLoopsIter(dataBdim,threads[1],threads[2])






totalFp,totalFn= 500,500
resList,resListIndicies= allocateResultLists(totalFp,totalFn)
iterNumb=3
isGold = 1
queueNumber = 11
xMeta,yMeta,zMeta = 1,1,1 # here by assumption that we have all meta indicies 0 based
isToBeValidated = true
resShmem = CUDA.zeros(Bool,(dataBdim[1]+2,dataBdim[2]+2,dataBdim[3]+2)) # we need this additional 33th an 34th spots

mainArrGPU = CuArray(mainArrCPU);
## IMPORTANT in that way we set all points to agree so all should be aded to results !
# refArrGPU= CuArray(mainArrCPU);
refArrGPU= CuArray(mainArrCPU);
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
loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ
inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(dataBdim[1] ,threads[1]),fld(dataBdim[2] ,threads[2]),dataBdim[3]    );
sourceShmem = CUDA.zeros(Bool,(dataBdim));
areToBeValidated= CUDA.ones(Bool,14)
isAnythingInPadding= CUDA.zeros(Bool,6)
function executeDataIterWithPaddingKernel(isAnythingInPadding,areToBeValidated,sourceShmem,inBlockLoopX,inBlockLoopY,inBlockLoopZ,resShmem,metaData,resList,resListIndicies,metaDataDims,refArrGPU,mainArrDims,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,dataBdim,mainArrGPU,iterNumb,isGold,xMeta,yMeta,zMeta,isToBeValidated)
    isMaskFull= true
    @executeDataIterWithPadding(mainArrDims, inBlockLoopX,inBlockLoopY,inBlockLoopZ,mainArrGPU,refArrGPU,xMeta,yMeta,zMeta,isGold,iterNumb)

    return
end

@cuda threads=threads blocks=blocks executeDataIterWithPaddingKernel(isAnythingInPadding,areToBeValidated,sourceShmem,inBlockLoopX,inBlockLoopY,inBlockLoopZ,resShmem,metaData,resList,resListIndicies,metaDataDims,refArrGPU,mainArrDims,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,dataBdim,mainArrGPU,iterNumb,isGold,xMeta,yMeta,zMeta,isToBeValidated)

metaCorrX = 2
metaCorrY = 2
metaCorrZ = 2
@test  metaData[metaCorrX,metaCorrY,metaCorrZ,getFullInSegmNumb()-1]==0
