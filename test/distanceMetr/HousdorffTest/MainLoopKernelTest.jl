using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
using ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils
using ..MainLoopKernel



##### clear before next dilatation
threads=(32,5);
blocks =1;
mainArrDims= (67,177,90);
dataBdim = (43,21,17);
metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);
metaDataDims= size(metaData);
resShmem = CUDA.ones(Bool,(dataBdim[1]+2,dataBdim[2]+2,dataBdim[3]+2 ));
sourceShmem = CUDA.ones(Bool,(dataBdim));
resShmemTotalLength = length(resShmem)
sourceShmemTotalLength= length(sourceShmem)
clearIterResShmemLoop= fld(resShmemTotalLength,threads[1]*threads[2])
clearIterSourceShmemLoop= fld(sourceShmemTotalLength,threads[1]*threads[2])
oldBlockCounter= CUDA.zeros(1)
function clearKernel(oldBlockCounter,resShmem,sourceShmem, clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength, sourceShmemTotalLength)
    locArr=0
    offsetIter=0
    # localOffset=0

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
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
using ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils
using ..MainLoopKernel

numberInWorkQueue = 2777
fpTotal=2000
fnTotal=1000
workQueaue = CUDA.zeros(Int64,4,4000)#allocateWorkQueue(fpTotal,fnTotal)
threads=(32,5);
blocks =17;
workQueauecounter= CuArray([numberInWorkQueue])
goldToBeDilatated= CUDA.ones(Bool,1)
segmToBeDilatated= CUDA.ones(Bool,1)
shmemSumLengthMaxDiv4= fld(36*14,4)*4
indexesArr = CUDA.zeros(Int64,1000000)
singleVal = CUDA.zeros(UInt32,1)

for i in 1:length(workQueaue)
    workQueaue[i]= i#using linear index
end
for i in 1 :(fpTotal+fnTotal)
    for j in 1:3
        workQueaue[j,i]= i*j
    end    
    workQueaue[4,i]=1 # always gold pass
end
# 70*4
# for i in 1:length(workQueaue)
#     workQueaue[i]= mod(i,255)#using linear index
# end
# for i in 1 :(fpTotal+fnTotal)
#     for j in 1:3
#         workQueaue[j,i]= mod(i*j,255)
#     end    
#     workQueaue[4,i]=1 # always gold pass
# end

# for i in 1:length(workQueaue)
#     workQueaue[i]=i#using linear index
# end
# for i in 1 :((fpTotal+fnTotal)*4)
#     workQueaue[i]=i
#     # for j in 1:3
#     #     workQueaue[j,i]=i*j
#     # end    
#     # workQueaue[4,i]=1 # always gold pass
# end
Int64.(Array(workQueaue[:,1:8]))

# workQueaue[*4+4]
function iterateOverWorkQueueKernel(indexesArr,workQueauecounter,workQueaue,goldToBeDilatated, segmToBeDilatated,shmemSumLengthMaxDiv4,singleVal)
    shmemSum =  @cuStaticSharedMem(UInt32,(36,14)) # we need this additional spots
    MainLoopKernel.@iterateOverWorkQueue(workQueauecounter,workQueaue,goldToBeDilatated, segmToBeDilatated,shmemSumLengthMaxDiv4,begin 

    @ifXY 1 1 CUDA.@atomic singleVal[1]+=1
    end) 

 return
end


@cuda threads=threads blocks=blocks iterateOverWorkQueueKernel(indexesArr,workQueauecounter,workQueaue,goldToBeDilatated, segmToBeDilatated,shmemSumLengthMaxDiv4,singleVal)
@test Int64(singleVal[1])==numberInWorkQueue

aa = (cld(33,2))
bb = shmemSumLengthMaxDiv4

fld((cld(33,2)),shmemSumLengthMaxDiv4)

(numberInWorkQueue*4)-sum(indexesArr)

coords = CartesianIndices(indexesArr[1:numberInWorkQueue*4])
aaa = sort(filter(coo-> indexesArr[coo]==0, coords)) #3128
