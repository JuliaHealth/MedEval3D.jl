####################3 idea is to have a test that covers all till the very very first work queue is established from the very begining
using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
using ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils,..ResultListUtils, ..Housdorff


mainArrDims= (60,60,60);
mainArrCPU= ones(UInt8,mainArrDims);
refArrCPU = ones(UInt8,mainArrDims);
##### we will create two planes 20 units apart from each 
mainArrCPU[10:50,10:50,10].= 1;
refArrCPU[10:50,10:50,30].= 1;

   
goldGPU = CuArray(mainArrCPU);
segmGPU= CuArray(mainArrCPU);

robustnessPercent= 0.95
numberToLooFor=1
boolKernelArgs, mainKernelArgs,threadsBoolKern,blocksBoolKern ,threadsMainKern,blocksMainKern,shmemSizeBool,shmemSizeMain= Housdorff.preparehousedorfKernel(goldGPU,segmGPU,robustnessPercent,numberToLooFor);
threadsBoolKern,Int64(blocksBoolKern) ,threadsMainKern,Int64(blocksMainKern)

goldGPU = CuArray(mainArrCPU);
segmGPU= CuArray(mainArrCPU);

function getHousedorffOnlyWorkQueue(goldGPUa,segmGPUa,boolKernelArgs,mainKernelArgs,threadsBoolKern,blocksBoolKern ,threadsMainKern,blocksMainKern,shmemSizeBool,shmemSizeMain)
    # boolKernelArgs[1]= goldGPU
    # boolKernelArgs[2]= segmGPU
   mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,loopXinPlane,loopYinPlane,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,numberToLooFor,inBlockLoopXZIterWithPadding,shmemblockDataLoop,shmemblockDataLenght,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength = boolKernelArgs

    @cuda threads=threadsBoolKern blocks=blocksBoolKern shmem=shmemSizeBool  cooperative=true boolKernelLoad(goldGPUa,segmGPUa,mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,numberToLooFor,inBlockLoopXZIterWithPadding,shmemblockDataLoop,shmemblockDataLenght,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength)
    #now time to get data structures dependent on bool kernel like for example loading subsections of meta data, creating work queue ...
    #some arrays needs to be instantiated only after we know the number of the false and true positives
    workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter= getBigGPUForHousedorffAfterBoolKernel(metaData,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp,reducedGoldA,reducedSegmA,dataBdim)
    dilatationArrsA,dilatationArrsB, mainArrDims,dataBdim ,numberToLooFor,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent   ,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter   ,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount   ,globalIterationNumb,workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter    ,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter   ,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter   ,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter   ,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed   ,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY   ,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp,resList,inBlockLoopXZIterWithPadding,paddingStore,shmemblockDataLenght,shmemblockDataLoop = mainKernelArgs

    dilatationArrsA= (reducedGoldA,reducedSegmA)
    dilatationArrs= (dilatationArrsA, dilatationArrsB)   
    #reverse order as when dilatating gold we want to establish do we cover segm voxels
    referenceArrs=(segmGPUa,goldGPUa )
       
    #main calculations
    @cuda threads=threadsMainKern blocks=blocksMainKern shmem=shmemSizeMain cooperative=true mainKernelLoadB( referenceArrs,dilatationArrs, mainArrDims,dataBdim
    ,numberToLooFor,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent
    ,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter
    ,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount
    ,globalIterationNumb,workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter
    ,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter
    ,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter
    ,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter
    ,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed
    ,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY
    ,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop
    ,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp,resList,inBlockLoopXZIterWithPadding,paddingStore,shmemblockDataLenght,shmemblockDataLoop)
        #@cuda threads=threadsMainKern blocks=blocksMainKern shmem=shmemSizeMain cooperative=true mainKernelLoad(dilatationArrs,referenceArrs, mainArrDims,dataBdim,numberToLooFor,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount,globalIterationNumb,workQueaue,resList,resListIndicies,maxResListIndex,inBlockLoopXZIterWithPadding,shmemblockDataLoop,shmemblockDataLenght,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp)
    return globalIterationNumb
    
    end


    """
    
    """
    function simpleWorkQueue()


    end    