using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils
using Main.MainLoopKernel



##### clear before next dilatation
threads=(32,5);
blocks =17;
mainArrDims= (67,177,90);
datBdim = (43,21,17);
metaData = MetaDataUtils.allocateMetadata(mainArrDims,datBdim);
metaDataDims= size(metaData);
resShmem = CUDA.ones(Bool,(datBdim[1]+2,datBdim[2]+2,datBdim[3]+2 ));
sourceShmem = CUDA.ones(Bool,(datBdim));
resShmemTotalLength = length(resShmem)
sourceShmemTotalLength= length(sourceShmem)
clearIterResShmemLoop= fld(resShmemTotalLength,threads[1]*threads[2])
clearIterSourceShmemLoop= fld(sourceShmemTotalLength,threads[1]*threads[2])
oldBlockCounter= CUDA.zeros(1)
function clearKernel(oldBlockCounter,resShmem,sourceShmem, clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength, sourceShmemTotalLength)
    locArr=0
    offsetIter=0
    localOffset=0

    MainLoopKernel.@clearBeforeNextDilatation( clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength, sourceShmemTotalLength)
 return
end


@cuda threads=threads blocks=blocks clearKernel(oldBlockCounter,resShmem,sourceShmem, clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength, sourceShmemTotalLength)
@test sum(resShmem)==0
@test sum(sourceShmem)==0


###########iterateOverWorkQueue
using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils
using Main.MainLoopKernel

numberInWorkQueue = 477
fpTotal=4
fnTotal=4
workQueaue = allocateWorkQueue(fpTotal,fnTotal)
threads=(32,5);
blocks =17;
workQueauecounter= CuArray([numberInWorkQueue])
goldToBeDilatated= CUDA.ones(Bool,1)
segmToBeDilatated= CUDA.ones(Bool,1)
shmemSumLengthMaxDiv4= fld(36*14,4)*4

singleVal = CUDA.zeros(UInt32,1)

for i in 1:length(workQueaue)
    workQueaue[i]= mod(i,255)#using linear index
end
for i in 1 :(fpTotal+fnTotal)
    for j in 1:3
        workQueaue[j,i]=i*j
    end    
    workQueaue[4,i]=1 # always gold pass
end
Int64.(Array(workQueaue[:,1:8]))

# workQueaue[*4+4]
function iterateOverWorkQueueKernel(workQueauecounter,workQueaue,goldToBeDilatated, segmToBeDilatated,shmemSumLengthMaxDiv4,singleVal)
    shmemSum =  @cuStaticSharedMem(UInt32,(36,14)) # we need this additional spots
    MainLoopKernel.@iterateOverWorkQueue(workQueauecounter,workQueaue,goldToBeDilatated, segmToBeDilatated,shmemSumLengthMaxDiv4,begin 

    @ifXY 1 1 @atomic singleVal[]+=1
    end) 

 return
end


@cuda threads=threads blocks=blocks iterateOverWorkQueueKernel(workQueauecounter,workQueaue,goldToBeDilatated, segmToBeDilatated,shmemSumLengthMaxDiv4,singleVal)
@test singleVal[1]==numberInWorkQueue


