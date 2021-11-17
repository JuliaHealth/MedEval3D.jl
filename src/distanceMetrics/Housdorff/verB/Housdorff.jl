
"""
collecting all needed  functions required to calculate Housdorff distance
"""
module Housdorff
using CUDA
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils,Main.ResultListUtils,Main.PrepareArrtoBool, Main.MainLoopKernel,Main.ScanForDuplicates
export getHousedorffDistance,boolKernelLoad,mainKernelLoad,get_shmemMainKernel,get_shmemBoolKernel,preparehousedorfKernel
"""
calculate housedorff distance of given arrays with given robustness percentage

"""
function getHousedorffDistance(goldGPUa,segmGPUa,boolKernelArgs,mainKernelArgs,threadsBoolKern,blocksBoolKern ,threadsMainKern,blocksMainKern)
    # boolKernelArgs[1]= goldGPU
    # boolKernelArgs[2]= segmGPU
    goldGPU,segmGPU,mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,numberToLooFor,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength = boolKernelArgs

    @cuda threads=threadsBoolKern blocks=blocksBoolKern cooperative=true boolKernelLoad(goldGPUa,segmGPUa,mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,numberToLooFor,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength)
    #now time to get data structures dependent on bool kernel like for example loading subsections of meta data, creating work queue ...
    #some arrays needs to be instantiated only after we know the number of the false and true positives
    metaData,reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB,workQueaue,resList,resListIndicies,maxResListIndex= getBigGPUForHousedorffAfterBoolKernel(metaData,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp,reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB,dataBdim)
       dilatationArrs= (reducedGoldA,reducedSegmA)
       referenceArrs= (reducedGoldB,reducedSegmB)
    dilatationArrs,referenceArrs, mainArrDims,dataBdim,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter,workQueaueCounter,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount,globalIterationNumb,workQueaue,resList,resListIndicies,maxResListIndex,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp = mainKernelArgs

    #main calculations
    @cuda threads=threadsMainKern blocks=blocksMainKern  cooperative=true mainKernelLoad(dilatationArrs,referenceArrs, mainArrDims,dataBdim,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter,workQueaueCounter,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount,globalIterationNumb,workQueaue,resList,resListIndicies,maxResListIndex,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp)

end

"""
for invoking getBoolCubeKernel
"""
function boolKernelLoad(goldGPU,segmGPU,mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,numberToLooFor,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength)
    @getBoolCubeKernel()
    return
end
"""
main function responsible for calculations of Housedorff distance
"""
function mainKernelLoad(dilatationArrs,referenceArrs, mainArrDims,dataBdim,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter,workQueaueCounter,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount,globalIterationNumb,workQueaue,resList,resListIndicies,maxResListIndex,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp)
    # resShmem =  @cuStaticSharedMem(Bool,(34,34,34  ))
    # sourceShmem =  @cuDynamicSharedMem(Bool,($dataBdim[1],$dataBdim[2],$dataBdim[3]))

    @mainLoopKernel()
    return
end


function get_shmemMainKernel(dataBdim)
    resShmem = cld((dataBdim[1]+2)*(dataBdim[2]+2)*(dataBdim[2]+2),8) #dividing by 8 as we want bytes
    sourceShmem = cld((dataBdim[1])*(dataBdim[2])*(dataBdim[2]),8) #dividing by 8 as we want bytes
    shmemSum= cld(36*14*32,8)
    areToBeValidated= cld(14,8)
    isAnythingInPadding= cld(6,8)
    alreadyCoveredInQueues= cld(32*14,8)
    someBools = 3
return resShmem+sourceShmem+shmemSum+areToBeValidated+isAnythingInPadding+alreadyCoveredInQueues+someBools
end

function get_shmemBoolKernel(threads)
    shmemSum= cld(32*32*2,8)
    minMaxes = 6
    localQuesValues = cld(32*14,8)
return shmemSum+minMaxes+localQuesValues
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

    loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength=calculateLoopsIter(dataBdim,threadsBoolKern[1],threadsBoolKern[2],metaDataDims,blocksBoolKern)
        minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp  =getSmallForBoolKernel();
        reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB=  getLargeForBoolKernel(mainArrDims,dataBdim);
    
    boolKernelArgs = (goldGPU,segmGPU,mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,numberToLooFor,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength)

    #### main kernel
    fpLoc = 100
    fnLoc = 100
    workQueaue= WorkQueueUtils.allocateWorkQueue(fpLoc,fnLoc)
    resList,resListIndicies,maxResListIndex= allocateResultLists(fpLoc,fnLoc)
    globalFpResOffsetCounter,globalFnResOffsetCounter,workQueaueCounter,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount,globalIterationNumb= getSmallGPUForHousedorff()

    dilatationArrs= (reducedGoldA,reducedSegmA)
    referenceArrs= (reducedGoldB,reducedSegmB)

    loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength = calculateLoopsIter(dataBdim,threadsMainKern[1],threadsMainKern[2],metaDataDims,blocksMainKern)
    shmemSumLengthMaxDiv4= fld((36*14),4)*4 # subject to futre changes
    mainKernelArgs= (dilatationArrs,referenceArrs, mainArrDims,dataBdim,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter,workQueaueCounter,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount,globalIterationNumb,workQueaue,resList,resListIndicies,maxResListIndex,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp)
    
    ## now we need to make use of occupancy API to get optimal number of threads and blocks fo each kernel
    threadsBoolKern,blocksBoolKern = getThreadsAndBlocksNumbForKernel(get_shmemBoolKernel,boolKernelLoad,boolKernelArgs)
    
    function get_shmemMainKernelLoc(threads)
        dataBdim = (threads[1],threads[2],32)
        get_shmemMainKernel(dataBdim)
    end

    threadsMainKern,blocksMainKern = getThreadsAndBlocksNumbForKernel(get_shmemMainKernel,mainKernelLoad,mainKernelArgs)

#now we get defoult values of data b dim  set on the basis of the threadsMainHKernel; and generally recalculating loops constants 
    dataBdim = (threadsMainKern[1],threadsMainKern[2],32)
    metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);
    metaDataDims= size(metaData);
    iterThrougWarNumb = cld(14,threadsMainKern[2])

    loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength=calculateLoopsIter(dataBdim,threadsBoolKern[1],threadsBoolKern[2],metaDataDims,blocksBoolKern)
    boolKernelArgs = (goldGPU,segmGPU,mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,numberToLooFor,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength)

    loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength=calculateLoopsIter(dataBdim,threadsBoolKern[1],threadsBoolKern[2],metaDataDims,blocksBoolKern)
    println("resShmemTotalLength $(resShmemTotalLength)  dataBdim $(dataBdim) \n")
    mainKernelArgs= (dilatationArrs,referenceArrs, mainArrDims,dataBdim,metaDataDims,metaData,iterThrougWarNumb,robustnessPercent,shmemSumLengthMaxDiv4,globalFpResOffsetCounter,globalFnResOffsetCounter,workQueaueCounter,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount,globalIterationNumb,workQueaue,resList,resListIndicies,maxResListIndex,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength, fn,fp)

return (boolKernelArgs, mainKernelArgs,threadsBoolKern,blocksBoolKern ,threadsMainKern,blocksMainKern )

end

end# module

    