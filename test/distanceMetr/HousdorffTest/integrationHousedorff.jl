
    ######### getting all together in Housedorff 
    # julia -g 2 C:\GitHub\GitHub\NuclearMedEval\test\distanceMetr\HousdorffTest\integrationHousedorff.jl
    using Revise, Parameters, Logging, Test
    using CUDA
    includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
    using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
    using ..MainLoopKernel,..PrepareArrtoBool,..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils,..ResultListUtils, ..Housdorff

    CUDA.allowscalar(true)

    mainArrDims= (120,120,120);
    mainArrCPU= zeros(UInt8,mainArrDims);
    refArrCPU = zeros(UInt8,mainArrDims);
    ##### we will create two planes 20 units apart from each 
    mainArrCPU[10:50,10:50,10].= 1;
    refArrCPU[10:50,10:50,80].= 1;
    # mainArrCPU[10:12,10:12,10].= 1;
    # refArrCPU[10:12,10:12,30].= 1;


        
    goldGPU = CuArray(mainArrCPU);
    segmGPU= CuArray(mainArrCPU);
    
    robustnessPercent= 0.95
    numberToLooFor=1
    boolKernelArgs, mainKernelArgs,threadsBoolKern,blocksBoolKern ,threadsMainKern,blocksMainKern ,shmemSizeBool,shmemSizeMain=    preparehousedorfKernel(goldGPU,segmGPU,robustnessPercent,numberToLooFor)
    mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,loopXinPlane,loopYinPlane,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,numberToLooFor,inBlockLoopXZIterWithPadding,shmemblockDataLoop,shmemblockDataLenght,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength=boolKernelArgs

   
   
        gold3d = CuArray(mainArrCPU)
        segm3d= CuArray(refArrCPU)
        numberToLooFor =UInt8(1)
        robustnessPercent = 0.9
   

    # globalIterationNumb= Housdorff.getHousedorffDistance(goldGPU,segmGPU,boolKernelArgs,mainKernelArgs,threadsBoolKern,blocksBoolKern ,threadsMainKern,blocksMainKern,shmemSizeBool,shmemSizeMain)
    # globalIterationNumb[1]




        boolKernelArgs, mainKernelArgs,threadsBoolKern,blocksBoolKern ,threadsMainKern,blocksMainKern ,shmemSizeBool,shmemSizeMain=    preparehousedorfKernel(gold3d,segm3d,robustnessPercent,numberToLooFor)
        mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,loopXinPlane,loopYinPlane,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,numberToLooFor,inBlockLoopXZIterWithPadding,shmemblockDataLoop,shmemblockDataLenght,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength=boolKernelArgs
   
        golddd = CuArray(mainArrCPU);
        segmmm= CuArray(refArrCPU);

    @cuda threads=threadsBoolKern blocks=blocksBoolKern shmem=shmemSizeBool  cooperative=true boolKernelLoad(golddd,segmmm,boolKernelArgs...)
    

    Int64(sum(mainArrCPU))
    Int64(fn[1])
    Int64(fp[1])
    Int64(minxRes[1])
    Int64(maxxRes[1])
    Int64( minyRes[1])
    Int64(maxyRes[1])
    Int64( minzRes[1])
    Int64(maxzRes[1])
size(reducedGoldA)
size(reducedSegmA)
size(metaData)
    #now time to get data structures dependent on bool kernel like for example loading subsections of meta data, creating work queue ...
    #some arrays needs to be instantiated only after we know the number of the false and true positives
 referenceArrs,dilatationArrs, mainArrDims,dataBdim,numberToLooFor,metaDataDims,metaDataNull,iterThrougWarNumb,robustnessPercent,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter    ,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount    ,globalIterationNumb,workQueue,workQueueCounter,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed    ,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop    ,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp,resList,inBlockLoopXZIterWithPadding,paddingStore,shmemblockDataLenght,shmemblockDataLoop= mainKernelArgs
        size(metaData)

    metaData,reducedGoldA  ,reducedSegmA ,paddingStore,resList,workQueue,workQueueCounter= getBigGPUForHousedorffAfterBoolKernel(metaData,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp,reducedGoldA,reducedSegmA,dataBdim)
    
    reducedSegmA[:,:,3]

    metaDataDims = size(metaData)
    metaDataLength= metaDataDims[1]*metaDataDims[2]*metaDataDims[3] 
    loopMeta= fld(metaDataLength,blocksMainKern )
    loopWarpMeta= cld(metaDataLength,(blocksMainKern*threadsMainKern[1] ))

    dilatationArrs= (reducedGoldA,reducedSegmA)
    #reverse order as when dilatating gold we want to establish do we cover segm voxels
    referenceArrs=(golddd,segmmm )
    shmemblockDataLenght = threadsMainKern[1]*threadsMainKern[2]*2
    shmemblockDataLoop = fld(shmemblockDataLenght,threadsMainKern[1]*threadsMainKern[2])
    
using  BenchmarkTools
    BenchmarkTools.DEFAULT_PARAMETERS.samples = 2
    BenchmarkTools.DEFAULT_PARAMETERS.seconds =600
    BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true
    

    #main calculations

    # @benchmark CUDA.@sync @cuda threads=threadsMainKern blocks=blocksMainKern shmem=shmemSizeMain cooperative=true mainKernelLoad( referenceArrs,dilatationArrs, mainArrDims,dataBdim
    @benchmark CUDA.@sync @cuda threads=threadsMainKern blocks=blocksMainKern shmem=shmemSizeMain cooperative=true mainKernelLoad( referenceArrs,dilatationArrs, mainArrDims,dataBdim
    ,numberToLooFor,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent
    ,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter
    ,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount
    ,globalIterationNumb,workQueue,workQueueCounter
    ,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed
    ,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY
    ,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop
    ,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp,resList,inBlockLoopXZIterWithPadding,paddingStore,shmemblockDataLenght,shmemblockDataLoop)
    


    globalIterationNumb[1]

    Int64(workQueueCounter[1])
    Int64.(workQueue)


#     (threadIdxX())+(innerWarpNumb+21)*33
#     32+(16+21)*33
# dataBdim[1]* dataBdim[2]*2



resList

for i in 1:16
    # println(" i $(i)  $((Int64(metaData[2,2,2, (getOldCountersBeg()+1)])))")
    println(" i $(i)  $((Int64(metaData[2,2,2, (getNewCountersBeg()+1)])))")
end
mainArrDims
