
"""
collecting all needed  functions required to calculate Housdorff distance
"""
module Housdorff
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils,Main.ResultListUtils

"""
calculate housedorff distance of given arrays with given robustness percentage
"""
function getHousedorffDistance()
   
   
    @cuda threads=threads blocks=blocks cooperative=true boolKernelLoad(mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,gold3d,segm3d,numberToLooFor,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta)
    #now time to get data structures dependent on bool kernel like for example loading subsections of meta data, creating work queue ...
    
       #some arrays needs to be instantiated only after we know the number of the false and true positives
       metaData,reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB,workQueaue,resList,resListIndicies= getBigGPUForHousedorffAfterBoolKernel(metaData,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp,reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB)
 
    #main calculations
    @cuda threads=threads blocks=blocks  cooperative=true mainKernelLoad()


    mainKernelLoad()
end

"""
for invoking getBoolCubeKernel
"""
function boolKernelLoad(mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,gold3d,segm3d,numberToLooFor,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta)
    @getBoolCubeKernel()
return
end
"""
main function responsible for calculations of Housedorff distance
"""
function mainKernelLoad()
    @mainLoopKernel()
end


function get_shmemMainKernel()
resShmem = cld((dataBdim[1]+2)*(dataBdim[2]+2)*(dataBdim[2]+2),8) #dividing by 8 as we want bytes
sourceShmem = cld((dataBdim[1])*(dataBdim[2])*(dataBdim[2]),8) #dividing by 8 as we want bytes
shmemSum= cld(36*14*32,8)
areToBeValidated= cld(14,8)
isAnythingInPadding= cld(6,8)
alreadyCoveredInQueues= cld(32*14,8)
someBools = 3
return resShmem+sourceShmem+shmemSum+areToBeValidated+isAnythingInPadding+alreadyCoveredInQueues+someBools
end

function get_shmemBoolKernel()
shmemSum= cld(32*32*2,8)
minMaxes = 6
localQuesValues = cld(32*14,8)
return shmemSum+minMaxes+localQuesValues
end

"""
creates required cu arrays , calculates some kernel constants and uses occupancy API
to calculate optimal number of threads and blocks to run a kernel
robustnessPercent - frequently we do not want to analyze all 
"""
function preparehousedorfKernel(goldGPU,segmGPU,robustnessPercent)
    metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);
    metaDataDims= size(metaData);
    #for bool cube kernel
    threadsBoolKern= (10,32); blocksBoolKern = 10#just some dummy will be modified after invoking occupancy API
    #for main kernel
    threadsMainKern= (10,32); blocksMainKern = 10#just some dummy will be modified after invoking occupancy API
    dataBdim = (32,32,32) # will be modified after number of threads gets calculated by occupancy API
    loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta=calculateLoopsIter(dataBdim,threads[1],threads[2],metaDataDims,blocks)
    
    loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta = calculateLoopsIter(dataBdim,threads[1],threads[2],metaDataDims,blocks)
        minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp  =getSmallForBoolKernel();
        reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB=  getLargeForBoolKernel(mainArrDims);
    
    boolKernelArgs = (mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,gold3d,segm3d,numberToLooFor,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta)

    mainKernelArgs= (dilatationArrs,referenceArrs, workQueauecounter,workQueaue,shmemSumLengthMaxDiv4)
## now we need to make use of occupancy API to get optimal number of threads and blocks fo each kernel
threadsBoollKernel,blocksBoollKernel = getThreadsAndBlocksNumbForKernel(get_shmemBoolKernel,kernelFun,args)
threadsMainHKernel,blocksMainHKernel = getThreadsAndBlocksNumbForKernel(get_shmemMainKernel,kernelFun,args)

end

end# module

    