"""
this will be invoked after first pass kernel so we already have a working queue  and we will work on it 
we will finish when mainActiveCounterNext will evaluate 0 at the end of iteration
as we will looop here intensively and after each iteration we will need to synchronize all blocks we will use cooperative goups and sync grid function
   
    reducedArrays =  [reducedSegmA,reducedGoldB,reducedGoldA,reducedSegmB] 
        - list of 3 dimensional boolean arrays with main data that we will work on

    metaData - basic data about data blocks - stored in 3 dimensional array
    metadataDims - dimensions of metaData array
    resArrays = [resArrayA, resArrayB]- 3 dimensional array of size of reduced source array where entry will be in place where we had covered the voxel, and value will mark in which itwration
        - we have two result arrays separate for each pass    
        - when we will cover some point  we need to report it in the resArray
    resArraysCounters= [resArrayAcounter, resArrayBcounter] - stores the numbers of results related to dilatation of each array - those will be needed to establish that 
                            - we can do early finish of the algorithm

    datablockDim - the edge length  describing the block of data we are working on  - defoult is 32 - which would produce blocks of size 32x32xzDim
        - we will have  the plane of threads of size 32x32 and in order to preserve memory coalescence we should have a warp have the  set of next x positions
    
    IMPORTANT we will alternate  between A nd B entries  and one will be used as working queue and other as a queue for next iteration    
    mainQuesCounterArr=[mainQuesCounterA,mainQuesCounterB] -array of two counters that we will update atomically and will be usefull to populate the work queue
    mainWorkQueueArr = [mainWorkQueueA,mainWorkQueueB]-2 element array of lists of the indicies of  data blocks in metadata with additional information is it referencing the goldpass or second one 

    isSecondCounter - Bool when true will mean that if true we should currently take data aout blocks wchich needs processed from second entry  
                        of mainQuesCounterArr and  mainWorkQueueArr and write activated blocks to the first ones - in case it is false - we do it in opposite mainQuesCounter
    tailCounter - variable accessed atomically by each block (if needed) and on each access increased - we will finish as this tail counter will equal mainQuesCounter 
    tailParts - number controlling how much of active blocks will be left to tail (part that is available to all blocks) - so the bigger this number the more of active blosk will go to 
                tail queue
    numberOfThreadBlocks - how many thread blocks we can squeeeze on our GPU  at once    
    isOddPass - boolean that will mark that we have an odd pass will be initialized to false as we had already first pass in other kernel  
                    will be important to point out which        mainQuesCounter and mainWorkQueue is currently a source of data and where we should put active and activated blocks
    iterationNumber - initialized to 1 and increased atomically after every dilation cycle (this variable is global)
    fp,fn - amount of false positives and false negatives - used in order to be able to get early termination
                    """
module MainLoopIterations
using KernelAbstractions, Logging, StaticArrays
using ..CUDAGpuUtils, ..HFUtils

@kernel function mainLoopIterationsKernel(
    reducedArrays,
    metaData,
    metadataDims::NTuple{3, UInt8},
    resArrays,
    resArraysCounters,
    datablockDim,
    mainQuesCounterArr,
    mainWorkQueueArr,
    tailCounter,
    tailParts::UInt8,
    numberOfThreadBlocks::UInt16,
    isOddPass,
    iterationNumber,
    fp,
    fn
)
    # needed to manage cooperative groups functions
    grid_handle = this_grid()
    # storing intermediate results +2 in order to get the one padding
    resShmem = @localmem(Bool, (34, 34, 34))
    # storing values loaded from analyzed array ...
    sourceShmem = @localmem(Bool, (34, 34, 34))
    # for storing sums for reductions
    shmemSum = @localmem(Float32, (35, 14))
    
    # coordinates of data in main array
    # we will use this to establish whether we should mark the data block as empty or full ...
    isMaskFull = false
    # here we will store in registers data uploaded from mask for later verification whether we should send it or not
    locArr = Int32(0)
    offsetIter = UInt16(0)

    # we will store here the data about the blocks that we want to process
    toIterWorkQueueShmem = @localmem(UInt8, (32, 4))
    # indicates where we are in general work queue in given moment if we are iterating over part of work queue owned by this thread block
    positionInLocalWorkQueaue = @localmem(UInt16, 1)

    # boolean useful in the iterating over private part of work queue
    isAnyBiggerThanZero = @localmem(Bool, 1)
    # used for iterating over a tail
    tailCounterInShmem = @localmem(UInt16, 1)
    workCounterInshmem = @localmem(UInt16, 1)

    # loading data to shared memory from global about is it even or odd pass
    isOddPassShmem = @localmem(Bool, 1)
    iterationNumberShmem = @localmem(UInt16, 1)
    # current spot in tail - useful to save where we need to access the tail to get proper block
    currentTailPosition = @localmem(UInt16, 1)
    # shared memory variables needed to mark whether we are already finished with any dilatation step
    goldToBeDilatated = @localmem(Bool, 1)
    segmToBeDilatated = @localmem(Bool, 1)
    # true when we have more than 0 blocks to analyze in next iteration
    workCounterBiggerThan0 = @localmem(Bool, 1)

    # resetting
    if @index(Local, Linear) == 0
        tailCounterInShmem[1] = 0
        workCounterInshmem[1] = 0
    end

    @synchronize

    # main loop logic
    while goldToBeDilatated[1] && segmToBeDilatated[1] && workCounterBiggerThan0[1]
        while isAnyBiggerThanZero[1]
            loadOwnedWorkQueueIntoShmem(mainWorkQueueArr, mainQuesCounterArr, toIterWorkQueueShmem, positionInLocalWorkQueaue, numberOfThreadBlocks, tailParts)
            # at this point if we have anything in the private part of the work queue we will have it in toIterWorkQueueShmem, in case we will find some 0 inside it means that queue is exhausted and we need to look into tail
            @unroll for i in UInt16(1):32 # most outer loop is responsible for z dimension
                if toIterWorkQueueShmem[i, 1] > 0
                    # we need to check also whether given dilatation step is already finished - for example it is possible that it does not make sense to dilate gold mask but still we need to dilate other
                    if (goldToBeDilatated[1] && toIterWorkQueueShmem[i, 4] == 1) || (segmToBeDilatated[1] && toIterWorkQueueShmem[i, 4] == 0)
                        innersingleDataBlockPass(toIterWorkQueueShmem[i, 4] == 1, toIterWorkQueueShmem[i, 1], toIterWorkQueueShmem[i, 2], toIterWorkQueueShmem[i, 3])
                    end
                end
            end
            loadDataNeededForTailAnalysisToShmem(currentTailPosition, tailCounter)
            @synchronize
        end
        # we set it before sync grid - as we do not need it for this one
        if @index(Local, Linear) == 0
            # we are just negating it so it should be the same in all blocks
            isOddPassShmem[1] = !isOddPassShmem[1]
        end
        @synchronize
        # if we are here it means we had covered all blocks that were marked as active and we need to prepare to next dilatation step
        sync_grid(grid_handle)
        # we clear and add negation to !isOddPassShmem because we want to set the previously updated counter to 0
        clearBeforeNextDilatation(locArr, resShmem, mainQuesCounterArr[!isOddPassShmem[1] + 1])
        prepareForNextDilatationStep(iterationNumber, tailCounter, numberOfThreadBlocks, tailParts, mainQuesCounterArr, isOddPassShmem)
    end
end

function executeMainLoopIterationsKernel(
    reducedArrays,
    metaData,
    metadataDims,
    resArrays,
    resArraysCounters,
    datablockDim,
    mainQuesCounterArr,
    mainWorkQueueArr,
    tailCounter,
    tailParts,
    numberOfThreadBlocks,
    isOddPass,
    iterationNumber,
    fp,
    fn
)
    threads = (32, 32)
    blocks = (cld(metadataDims[1], threads[1]), cld(metadataDims[2], threads[2]), cld(metadataDims[3], threads[3]))

    kernel = mainLoopIterationsKernel(CPU(), threads, blocks)
    kernel(
        reducedArrays,
        metaData,
        metadataDims,
        resArrays,
        resArraysCounters,
        datablockDim,
        mainQuesCounterArr,
        mainWorkQueueArr,
        tailCounter,
        tailParts,
        numberOfThreadBlocks,
        isOddPass,
        iterationNumber,
        fp,
        fn,
        ndrange = blocks
    )
end

"""
checking weather we already finished so we need to check    
    - is amount of results related to gold mask dilatations is equal to false positives
    - is amount of results related to other  mask dilatations is equal to false negatives
    - is amount of workQueue that we will want to analyze now is bigger than 0 
"""
function isNotFinished(resArraysCounters,fp,fn,workCounterInshmem,goldToBeDilatated,segmToBeDilatated)::Bool
   
    resArraysCounters[2] == fp[1]
    resArraysCounters[1] == fn[1]



end#isNotFinished

"""
clearing data in order to enable their reuse in next iteration
"""
function clearBeforeNextDilatation(locArr,resShmem,oldBlockCounter)
    #resetting the counter so when we will add to this new items it will count from 0 
    if(threadIdxX()==7 && threadIdxY()==7)
        oldBlockCounter[1]=0
    end    

    #probably we do not need to clear work queue as we will just overwrite it
    #workQueue[((blockDimX()-1)*1024)+ (((threadIdxX()-1)*32)+1)+threadIdxY()]
    clearLocArr(locArr)
    clearMainShmem(resShmem)
    clearPadding(resShmem)
end#clearBeforeNextDilatation


"""
before next loop after grid sync we need to 
    - increase iteration Number
    - set tail counter to correct value (1 bigger than last value from private queue of last block)
those should be invoked only once per grid hence it will  be invoked only in first thread block (first chosen arbitrarly)
"""
function prepareForNextDilatationStep(iterationNumber,tailCounter,numberOfThreadBlocks,tailParts,mainQuesCounterArr,isOddPassShmem)
    if(blockIdx().x==1)  
        @IfYX 8 8 iterationNumber[1]= CUDA.atomic_inc!(pointer(iterationNumber), UInt16(1))+1
        @IfYX 9 9 tailCounter[1]=cld(mainQuesCounterArr[isOddPassShmem[1]+1],(numberOfThreadBlocks+tailParts))*numberOfThreadBlocks +1
    end#if
end#prepareForNextDilatationStep



"""
load data needed for tail analysis into shared memory
"""
function loadDataNeededForTailAnalysisToShmem(currentTailPosition,tailCounter )
    @IfYX 4 4 currentTailPosition[1]= CUDA.atomic_inc!(pointer(tailCounter), UInt16(1))+1
end#loadDataNeededForTailAnalysisToShmem   


"""
loads data at the begining of each dilatation step
    we need to set some variables in shared memory ro initial values
"""
function loadDataAtTheBegOfDilatationStep(isOddPassShmem,iterationNumberShmem,iterationNumber,positionInMainWorkQueaue,workCounterInshmem,mainQuesCounterArr,isAnyBiggerThanZero,goldToBeDilatated,segmToBeDilatated, resArraysCounters  )

    @IfYX 2 2 iterationNumberShmem[1]= iterationNumber[1]
    @IfYX 3 3 positionInMainWorkQueaue[1]=0 
    @IfYX 4 4 begin
        workCounterInshmem[1]= mainQuesCounterArr[isOddPassShmem[1]+1]
        workCounterBiggerThan0[1]= (workCounterInshmem[1]>0)
                    end 
    @IfYX 5 5 isAnyBiggerThanZero[]=true  
    @IfYX 6 6 goldToBeDilatated[1]=(resArraysCounters[2] < fp[1])
    @IfYX 7 7  segmToBeDilatated[1]=(resArraysCounters[1] < fn[1])

end

"""
load data from current queue (part related to this block) to shared memory  so we can iterate over effectively
    we can load no more than 32 data points at a time
    
    mainWorkQueue - data we are currently iterating over - list of the indicies of  data blocks in metadata with additional information is it referencing the goldpass or second one  
    mainQuesCounter - wil give us information how many active blocks we have in our work queue 
    positionInLocalWorkQueaue- indicates where we are in locla shared memory work queue in given moment if we are iterating over part of work queue owned by this thread block
    toIterWorkQueueShmem - we will store here the data about the blocks that we want to process 
    numberOfThreadBlocks - how many thread blocks we can squeeeze on our GPU  at once          
    tailParts - number controlling how much of active blocks will be left to tail (part that is available to all blocks) - so the bigger this number the more of active blosk will go to 
                tail queue

"""
function loadOwnedWorkQueueIntoShmem(mainWorkQueue
                                    ,mainQuesCounter
                                    ,toIterWorkQueueShmem
                                    ,positionInMainWorkQueaue
                                    ,numberOfThreadBlocks
                                    ,tailParts)


 privatePartLength=cld(mainQuesCounter[1],(numberOfThreadBlocks+tailParts))
 localPosition= blockIdx().x *privatePartLength + positionInMainWorkQueaue[1]+threadIdxY()
    #TODO() check for coalescence
    #as we want to load only 32 at a time we use 1 warp
    if(threadIdxX()==1)
        toIterWorkQueueShmem[threadIdxY(),:]= [UInt8(0),UInt8(0),UInt8(0),UInt8(0)]# so it is 0 and if we will not go through loop below it will remain 0 
    end    
    if(threadIdxX()==1 &&  localPosition < ((blockIdx().x+1) *privatePartLength) )
    toIterWorkQueueShmem[threadIdxY(),:]=mainWorkQueue[localPosition]#we add short 4 element arrays here
    end 
    #now we need to mark that we loaded 32  elemnts
    
    positionInLocalWorkQueaue[1]+=32
 
end    



"""
collects multiple functions that will be invoked over a single data blocks
analyzedArr - array that we dilatate
refAray - array we are comparing with
iterationNumber - in first pass it is 1 
blockBeginingX,blockBeginingY,blockBeginingZ - coordinates where our block is begining - will be used as offset by our threads
isMaskFull,isMaskEmpty - needed for establishing state  of a current block after processing 
resShmem- shared memory
locArr - thread local array with values loaded from global memory
metaData - 3 dim array with metadata of data blocks
metadataDims - dimensions of metadataDims
currMatadataBlockX,currMatadataBlockY, currMatadataBlockZ - cartesian coordinates of current block in metadaa!!!

IMPORTANT mainQuesCounter and mainWorkQueue will alternate as we will have 2 of them and every iteration we will swith between one to another
mainQuesCounter - counter that we will update atomically and will be usefull to populate the work queue
mainWorkQueue - the list of the indicies of  data blocks in metadata with additional information is it referencing the goldpass or second one 
resArraysCounter - counter needed to add keep track on how many results we have in our result array
                    - we have separate counter for gold pass and other pass
"""
function singleDataBlockPass(analyzedArr
                ,refAray
                ,iterationNumber
                ,blockBeginingX
                ,blockBeginingY
                ,blockBeginingZ
                ,isMaskFull
                ,resShmem
                ,locArr
                ,metaData
                ,metadataDims::Tuple{UInt8,UInt8,UInt8}
                ,isPassGold::Bool
                ,currMatadataBlockX::UInt8
                ,currMatadataBlockY::UInt8
                ,currMatadataBlockZ::UInt8
                ,mainQuesCounter
                ,mainWorkQueue
                ,resArray
                ,resArraysCounter )

            #clear shared memory 
            clearMainShmem(resShmem,32)
            clearPadding(resShmem,32)# we separately clear padding
            ############### execution
            executeDataIterOtherPasses(analyzedArr, refAray,iterationNumber,blockBeginingX,blockBeginingY,blockBeginingZ,isMaskFull,resShmem,locArr,resArraysCounter)
            #for futher processing we need to have space in main shmem
            clearMainShmem(shmem)
            #now we need to deal with padding in shmem res
            processAllPaddingPlanes(x,y,z,shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArray,metaData,metadataDims,isPassGold,locArr,resArraysCounter)
            # now let's check weather block is eligible for futher processing - for this we need sums ...
            isActiveForNormalPass(isMaskFull, isMaskEmpty,resShmem,currMatadataBlockX,currMatadataBlockY,currMatadataBlockZ,isPassGold,metaData,mainQuesCounter,mainWorkQueue,resArraysCounter)
end#singleDataBlockPass
end