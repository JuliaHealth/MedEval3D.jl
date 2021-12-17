
"""
collecting all needed  functions required to calculate Housdorff distance
"""
module Housdorff
using CUDA
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.BitWiseUtils,Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils,Main.ResultListUtils,Main.PrepareArrtoBool, Main.MainLoopKernel,Main.ScanForDuplicates
export get_shmemMainKernel, getHousedorffDistance,boolKernelLoad,mainKernelLoad,get_shmemMainKernel,get_shmemBoolKernel,preparehousedorfKernel
"""
calculate housedorff distance of given arrays with given robustness percentage

"""
function getHousedorffDistance(goldGPUa,segmGPUa,boolKernelArgs,mainKernelArgs,threadsBoolKern,blocksBoolKern ,threadsMainKern,blocksMainKern,shmemSizeBool,shmemSizeMain)
    # boolKernelArgs[1]= goldGPU
    # boolKernelArgs[2]= segmGPU
   mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,loopXinPlane,loopYinPlane,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,numberToLooFor,inBlockLoopXZIterWithPadding,shmemblockDataLoop,shmemblockDataLenght,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength = boolKernelArgs

    @cuda threads=threadsBoolKern blocks=blocksBoolKern shmem=shmemSizeBool  cooperative=true boolKernelLoad(goldGPUa,segmGPUa,mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,numberToLooFor,inBlockLoopXZIterWithPadding,shmemblockDataLoop,shmemblockDataLenght,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength)
    #now time to get data structures dependent on bool kernel like for example loading subsections of meta data, creating work queue ...
    #some arrays needs to be instantiated only after we know the number of the false and true positives
    workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter= getBigGPUForHousedorffAfterBoolKernel(metaData,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp,reducedGoldA,reducedSegmA,dataBdim)
    dilatationArrsA,dilatationArrsB, mainArrDims,dataBdim ,numberToLooFor,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent   ,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter   ,workQueaueCounter,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount   ,globalIterationNumb,workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter    ,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter   ,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter   ,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter   ,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed   ,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY   ,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp,resList = mainKernelArgs

    dilatationArrsA= (reducedGoldA,reducedSegmA)
    dilatationArrsB= (reducedGoldB,reducedSegmB)
    dilatationArrs= (dilatationArrsA, dilatationArrsB)   
    #reverse order as when dilatating gold we want to establish do we cover segm voxels
    referenceArrs=(segmGPUa,goldGPUa )
       
    #main calculations
    @cuda threads=threadsMainKern blocks=blocksMainKern shmem=shmemSizeMain cooperative=true mainKernelLoadB( referenceArrs,dilatationArrs, mainArrDims,dataBdim
    ,numberToLooFor,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent
    ,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter
    ,workQueaueCounter,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount
    ,globalIterationNumb,workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter
    ,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter
    ,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter
    ,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter
    ,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed
    ,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY
    ,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop
    ,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp,resList)
        #@cuda threads=threadsMainKern blocks=blocksMainKern shmem=shmemSizeMain cooperative=true mainKernelLoad(dilatationArrs,referenceArrs, mainArrDims,dataBdim,numberToLooFor,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter,workQueaueCounter,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount,globalIterationNumb,workQueaue,resList,resListIndicies,maxResListIndex,inBlockLoopXZIterWithPadding,shmemblockDataLoop,shmemblockDataLenght,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp)
    return globalIterationNumb
    
    end

"""
clearing data before next execution
"""
function housClearForExecution()

end


"""
for invoking getBoolCubeKernel
"""
function boolKernelLoad(goldGPU,segmGPU,mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,loopXinPlane,loopYinPlane,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,numberToLooFor,inBlockLoopXZIterWithPadding,shmemblockDataLoop,shmemblockDataLenght,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength)
    @getBoolCubeKernel()
    return
end
"""
main function responsible for calculations of Housedorff distance
"""
function mainKernelLoadB(referenceArrs,dilatationArrs, mainArrDims,dataBdim
    ,numberToLooFor,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent
    ,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter
    ,workQueaueCounter,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount
    ,globalIterationNumb,workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter
    ,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter
    ,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter
    ,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter
    ,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed
    ,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY
    ,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop
    ,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp,resList)

    @mainLoopKernel()
    return
end
function mainKernelLoad(referenceArrs,dilatationArrs, mainArrDims,dataBdim
    ,numberToLooFor,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent
    ,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter
    ,workQueaueCounter,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount
    ,globalIterationNumb,workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter
    ,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter
    ,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter
    ,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter
    ,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed
    ,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY
    ,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop
    ,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp,resList)
    @mainLoopKernel()
    return
end


function get_shmemMainKernel(dataBdim)
    
    shmemblockData= sizeof(UInt32)*dataBdim[1]* dataBdim[2]
    resShmemblockData= sizeof(UInt32)*dataBdim[1]* dataBdim[2]
    shmemPaddings= sizeof(Bool)*(  max(dataBdim[1], dataBdim[2]))*(  max(dataBdim[1], dataBdim[2]))*6
    shmemSum= sizeof(UInt32)*36*14
    areToBeValidated= sizeof(Bool)*14
    isAnythingInPadding= sizeof(Bool)*7
    alreadyCoveredInQueues= sizeof(UInt32)*14
    someBools = sizeof(Bool)*4
    someInt16 = sizeof(UInt16)*3
    workCountersInshmem = sizeof(UInt16)*8
return workCountersInshmem+shmemSum+areToBeValidated+isAnythingInPadding+alreadyCoveredInQueues+someBools+shmemblockData+someInt16+resShmemblockData+shmemPaddings
end

function get_shmemBoolKernel(dataBdim)
    # shmemSum= sizeof(Float32)*32*2
    shmemblockData= sizeof(Int32)*(dataBdim[1])*(dataBdim[2])
    localQuesValues = sizeof(UInt32)*14
    minMaxes = sizeof(Float32)*6
return minMaxes+localQuesValues+shmemblockData
end

"""
creates required cu arrays , calculates some kernel constants and uses occupancy API
to calculate optimal number of threads and blocks to run a kernel
robustnessPercent - frequently we do not want to analyze all of the fap and fn in order to reduce the impact of the outliers  
numberToLooFor - what we will look for in main arrays
"""
function preparehousedorfKernel(goldGPU,segmGPU,robustnessPercent,numberToLooFor)
    mainArrDims = size(goldGPU)
    dataBdim = (32,32,32) # will be modified after number of threads gets calculated by occupancy API

    metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);
    metaDataDims= size(metaData);
    #for bool cube kernel
    threadsBoolKern= (30,32); blocksBoolKern = 10#just some dummy will be modified after invoking occupancy API
    #for main kernel
    threadsMainKern= (30,32); blocksMainKern = 10#just some dummy will be modified after invoking occupancy API
    iterThrougWarNumb = cld(14,threadsMainKern[2])

    inBlockLoopXZIterWithPadding,shmemblockDataLoop,shmemblockDataLenght,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength=calculateLoopsIter(dataBdim,threadsBoolKern[1],threadsBoolKern[2],metaDataDims,blocksBoolKern)
        minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp  =getSmallForBoolKernel();
        reducedGoldA,reducedSegmA=  getLargeForBoolKernel(mainArrDims,dataBdim);
   
    loopXinPlane,loopYinPlane = 1,1
    boolKernelArgs = (mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,loopXinPlane,loopYinPlane,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,numberToLooFor,inBlockLoopXZIterWithPadding,shmemblockDataLoop,shmemblockDataLenght,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength)
    #needed also in bool kernel
    
    #### main kernel
    fpLoc = 10
    fnLoc = 10
    resList= allocateResultLists(fpLoc,fnLoc)
    globalFpResOffsetCounter,globalFnResOffsetCounter,workQueaueCounter,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount,globalIterationNumb= getSmallGPUForHousedorff()

    dilatationArrsA= (reducedGoldA,reducedSegmA)


    inBlockLoopXZIterWithPadding,shmemblockDataLoop,shmemblockDataLenght,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength=calculateLoopsIter(dataBdim,threadsBoolKern[1],threadsBoolKern[2],metaDataDims,blocksBoolKern)
    shmemSumLengthMaxDiv4= fld((36*14),4)*4 # subject to futre changes
    metaData,reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB,resList,workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter= getBigGPUForHousedorffAfterBoolKernel(metaData,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp,reducedGoldA,reducedSegmA,dataBdim)

    mainKernelArgs= (dilatationArrs, mainArrDims,dataBdim
    ,numberToLooFor,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent
    ,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter
    ,workQueaueCounter,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount
    ,globalIterationNumb,workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter
    ,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter
    ,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter
    ,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter
    ,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed
    ,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY
    ,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop
    ,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp,resList)
   
    function get_shmemMainKernelLoc(threads)
        dataBdim = (threads[1],threads[2],32)
        get_shmemMainKernel(dataBdim)
    end
    

    threadsMainKern,blocksMainKern = getThreadsAndBlocksNumbForKernel(get_shmemMainKernel,mainKernelLoad,(goldGPU,segmGPU,mainKernelArgs...))
    function get_shmemBoolKernelLoc(threads)
        dataBdim = (threadsMainKern[1],threadsMainKern[2],32)
        get_shmemBoolKernel(dataBdim)
    end
     ## now we need to make use of occupancy API to get optimal number of threads and blocks fo each kernel
    threadsBoolKern,blocksBoolKern = getThreadsAndBlocksNumbForKernel(get_shmemBoolKernelLoc,boolKernelLoad,(goldGPU,segmGPU,boolKernelArgs...))
    
    loopXinPlane,loopYinPlane = fld(threadsMainKern[1],threadsBoolKern[1] ), fld(threadsMainKern[2],threadsBoolKern[2] )
#now we get defoult values of data b dim  set on the basis of the threadsMainHKernel; and generally recalculating loops constants 
    dataBdim = (threadsMainKern[1],threadsMainKern[2],32)
    metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);
    metaDataDims= size(metaData);
    iterThrougWarNumb = cld(14,threadsMainKern[2])

    boolKernelArgs = (mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,loopXinPlane,loopYinPlane,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,numberToLooFor,inBlockLoopXZIterWithPadding,shmemblockDataLoop,shmemblockDataLenght,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength)

    inBlockLoopXZIterWithPadding,shmemblockDataLoop,shmemblockDataLenght,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength=calculateLoopsIter(dataBdim,threadsBoolKern[1],threadsBoolKern[2],metaDataDims,blocksBoolKern)
    
    boolKernelArgs = (mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,loopXinPlane,loopYinPlane,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,numberToLooFor,inBlockLoopXZIterWithPadding,shmemblockDataLoop,shmemblockDataLenght,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength)
    metaData,reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB,resList,workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter= getBigGPUForHousedorffAfterBoolKernel(metaData,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp,reducedGoldA,reducedSegmA,dataBdim)

    mainKernelArgs= (dilatationArrs, mainArrDims,dataBdim
    ,numberToLooFor,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent
    ,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter
    ,workQueaueCounter,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount
    ,globalIterationNumb,workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter
    ,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter
    ,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter
    ,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter
    ,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed
    ,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY
    ,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop
    ,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp,resList)    
    shmemSizeBool=get_shmemBoolKernelLoc(threadsBoolKern)
    shmemSizeMain=get_shmemMainKernelLoc(threadsMainKern)

    CUDA.unsafe_free!(goldGPU)
    CUDA.unsafe_free!(segmGPU)
    #CUDA.reclaim()
return (boolKernelArgs, mainKernelArgs,threadsBoolKern,blocksBoolKern ,threadsMainKern,blocksMainKern ,shmemSizeBool,shmemSizeMain)

end

end# module

    
