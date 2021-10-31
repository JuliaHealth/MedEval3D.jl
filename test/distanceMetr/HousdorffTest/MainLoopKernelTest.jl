using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils
using Main.MainLoopKernel



##### iter data block
threads=(32,5);
blocks =17;
mainArrDims= (67,177,90);
datBdim = (43,21,17);
mainArrCPU= falses(mainArrDims);
mainArrCPU[5,5,5]= true;
mainArrGPU = CuArray(mainArrCPU);
metaData = MetaDataUtils.allocateMetadata(mainArrDims,datBdim);
metaDataDims= size(metaData);
inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(datBdim[1] ,threads[1]),fld(datBdim[2] ,threads[2]),datBdim[3]    );
#we are iterating here block by block sequentially
loopXMeta,loopYZMeta= (1,1)#(metaDataDims[1],fld(metaDataDims[2]*metaDataDims[3] ,blocks)  )

resShmem = CUDA.zeros(Bool,(datBdim[1]+2,datBdim[2]+2,datBdim[3]+2 ));
sourceShmem = CUDA.zeros(Bool,(datBdim));

function processDataKernel(mainArrGPU,sourceShmem,resShmem,mainArrDims,metaDataDims,loopXMeta,loopYZMeta,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ )
    @iter3dOuter(metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,
    begin 
        #@ifXY 1 1    CUDA.@cuprint "  xMeta $(xMeta) yMeta $(yMeta)  zMeta $(zMeta) \n"   

        @iterDataBlock(mainArrDims,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,
        begin
            maskBool=mainArrGPU[x,y,z]
            ProcessMainDataVerB.@processMaskData( maskBool)

    end)end)
    
    return
end


@cuda threads=threads blocks=blocks processDataKernel(mainArrGPU,sourceShmem,resShmem,mainArrDims,metaDataDims,loopXMeta,loopYZMeta,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ )
@test  resShmem[5+1,5+1,5+1]==false