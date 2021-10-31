
"""
main kernel managing 
"""
function mainLoopKernel()

    @mainLoopKernelAllocations()
    loadDataAtTheBegOfDilatationStep(true,iterationNumberShmem,iterationNumber,positionInMainWorkQueaue,workCounterInshmem,mainQuesCounterArr,isAnyBiggerThanZero,goldToBeDilatated,segmToBeDilatated, resArraysCounters  )
    sync_threads()
    #we check first wheather next dilatation step should be done or not we also establish some shared memory variables to know wheather both passes should continue or just one
    # checking weather we already finished so we need to check    
    # - is amount of results related to gold mask dilatations is equal to false positives
    # - is amount of results related to other  mask dilatations is equal to false negatives
    # - is amount of workQueue that we will want to analyze now is bigger than 0 
    while(goldToBeDilatated[1] && segmToBeDilatated[1] && workCounterBiggerThan0[1])
        @mainLoop()
    end#while we did not yet finished    


end



"""
main loop logic
"""
macro mainLoop()
    loadOwnedWorkQueueIntoShmem(mainWorkQueue,mainQuesCounter,toIterWorkQueueShmem,positionInMainWorkQueaue ,numberOfThreadBlocks,tailParts)
    #at this point if we have anything in the private part of the  work queue we will have it in toIterWorkQueueShmem, in case  we will find some 0 inside it means that queue is exhousted and we need to loo into tail
    #we are analyzing here data  from private part of work queue
    while(isAnyBiggerThanZero[]) #isAnyBiggerThanZero will be modified inside @privateWorkQueueAnalysis
        @privateWorkQueueAnalysis()
    end#while
    @IfYX 4 4 currentTailPosition[1]= CUDA.atomic_inc!(pointer(tailCounter), UInt16(1))+1

    # in this moment we have nothing left in private part of work queue and we need to check is there sth in tail to process
    #first load data of tail counter once

    sync_threads()
    while (currentTailPosition[1]< workCounterInshmem[1])
        @analyzeTail()
    end

    #we are just negiting it so it should be the same in all blocks
    @IfXY 1 1 isOddPassShmem[1]= !isOddPassShmem[1]
    #if we are here it means we had covered all blocks that were marked as active and we need to prepare to next dilatation step

    sync_grid(grid_handle)
        @prepareForNextDilation()
    sync_grid(grid_handle)

end#mainLoop


"""
allocates memory in the kernel some register and shared memory  (no allocations in global memory here from kernel)
"""
macro mainLoopKernelAllocations()
 #needed to manage cooperative groups functions
 grid_handle = this_grid()
 #storing intermidiate results +2 in order to get the one padding 
 resShmem =  @cuStaticSharedMem(Bool,(34,34,34))
 #storing values loaded from analyzed array ...
 sourceShmem =  @cuStaticSharedMem(Bool,(32,32,32))
 #for storing sums for reductions
 shmemSum =  @cuStaticSharedMem(UInt32,(36,14)) # we need this additional spots
# used to accumulate counts from of fp's and fn's already covered in dilatation steps
alreadyCoveredInQueues =@cuStaticSharedMem(UInt32,(14))
 
 #coordinates of data in main array
 #we will use this to establish weather we should mark  the data block as empty or full ...
 isMaskFull= false
 #here we will store in registers data uploaded from mask for later verification wheather we should send it or not
 locArr= Int32(0)
 offsetIter = UInt16(0)

# #used in  meta data analyze pass in order to mark later is there anything for validation still in a whole image - later it would be passed into global memory as information from all blocks ...
# sthToBeValidatedLocalFp = @cuStaticSharedMem(Bool,1)
# sthToBeValidatedLocalFn = @cuStaticSharedMem(Bool,1)


 #we will store here the data about the blocks that we want to process 
 toIterWorkQueueShmem =  @cuStaticSharedMem(UInt8,32,4)
 #indicates where we are in general work queue in given moment if we are iterating over part of work queue owned by this thread block
 positionInLocalWorkQueaue =  @cuStaticSharedMem(UInt16,1)
 #boolean usefull in the iterating over private part of work queue 
 isAnyBiggerThanZero =  @cuStaticSharedMem(Bool,1)
 #used for iterating over a tail
 tailCounterInShmem= @cuStaticSharedMem(UInt16,1)
 workCounterInshmem= @cuStaticSharedMem(UInt16,1)

 #loading data to shared memory from global about is it even or odd pass
 isOddPassShmem =  @cuStaticSharedMem(Bool,1)
 iterationNumberShmem =  @cuStaticSharedMem(UInt16,1)
 #current spot in tail - usefull to save where we need to access the tail to get proper block
 currentTailPosition =  @cuStaticSharedMem(UInt16,1)
 #shared memory variables needed to marks wheather we are already finished with  any dilatation step
 goldToBeDilatated =  @cuStaticSharedMem(Bool,1)
 segmToBeDilatated =  @cuStaticSharedMem(Bool,1)
 #true when we have more than 0 blocks to analyze in next iteration
 workCounterBiggerThan0 =  @cuStaticSharedMem(Bool,1) 
 @IfYX 1 1 iterationNumberShmem[1]= 0
 sync_threads()
end    

"""
stores the most important part of the kernel where we analyze supplied data blocks
do the dilatation and add to the result queue
"""
macro  innersingleDataBlockPass()
            #in order to be able to skip some of the validations we will load now informations about this block and neighbouring blocks 
           #like for example are there any futher results to be written in this block including border queues
           #and is there sth in border queues of the neighbouring data blocks
           @fillPreValidationsCheck()

           sync_threads()
           ############### execution
           @executeDataIterWithPadding() 

end

"""
clearing source shared memory
"""
function clearSourceShmem(sourceShmem)
    krowa
end#clearSourceShmem

"""
clears shmem sum taht is used for reductions
"""
function clearShmemSum(shmemSum)
    krowa
end
"""
in order to be able to skip some of the validations we will load now informations about this block and neighbouring blocks 
like for example are there any futher results to be written in this block including border queues
and is there sth in border queues of the neighbouring data blocks
problem is in case of the corners  as becouse of corners it may be included in diffrent resuld queues

"""
macro fillPreValidationsCheck()
    #we keep function in metadata utils
    getIstoBeAnalyzed(resShmem,metaData,linIndex,isGold)

end#fillPreValidationsCheck



"""
analyzing private part of work queue
"""
macro privateWorkQueueAnalysis()
    loadOwnedWorkQueueIntoShmem(mainWorkQueue,mainQuesCounter,toIterWorkQueueShmem,positionInMainWorkQueaue ,numberOfThreadBlocks,tailParts)
    #at this point if we have anything in the private part of the  work queue we will have it in toIterWorkQueueShmem, in case  we will find some 0 inside it means that queue is exhousted and we need to loo into tail
    @unroll for i in UInt16(1):32# most outer loop is responsible for z dimension
        if(toIterWorkQueueShmem[i,1]>0)
            #we need to check also wheather given dilatation step is already finished - for example it is possible that it do not make sense to dilatate gold mask but still we need dilate other
            if( (goldToBeDilatated[1]&&toIterWorkQueueShmem[i,4]==1) || (segmToBeDilatated[1]&&toIterWorkQueueShmem[i,4]==0)  )    
                @innersingleDataBlockPass(toIterWorkQueueShmem[i,4]==1,toIterWorkQueueShmem[i,1],toIterWorkQueueShmem[i,2] ,toIterWorkQueueShmem[i,3] )
            end
            sync_threads()
        else    #at this point we got some 0 in the toIterWorkQueueShmem 
            @ifXY 1 1  isAnyBiggerThanZero[]=false
            sync_threads()
        end#if    
    end#for    
end    


"""
analyzing tail part of work queue
"""
macro analyzeTail()
    #below we access tail of working queue in a way that will be atomic
    @ifXY 1 1 toIterWorkQueueShmem[1]= mainWorkQueueArr[isOddPassShmem[]+1,currentTailPosition[1]]
    sync_threads()
    #we need to check also wheather given dilatation step is already finished - for example it is possible that it do not make sense to dilatate gold mask but still we need dilate other
    if( (goldToBeDilatated[1]&&toIterWorkQueueShmem[1,4]==1) || (segmToBeDilatated[1]&&toIterWorkQueueShmem[1,4]==0)  )    
        @innersingleDataBlockPass(toIterWorkQueueShmem[1,4]==1,toIterWorkQueueShmem[1,1],toIterWorkQueueShmem[1,2] ,toIterWorkQueueShmem[1,3] )
    end
    loadDataNeededForTailAnalysisToShmem(currentTailPosition,tailCounter )
    sync_threads()
end




"""
after dilatation prepare
"""
macro prepareForNextDilation()
    #we update metadata and prepare work queue
    setMEtaDataOtherPasses()
    #we clear  and add negation to !isOddPassShmem becouse we want to set the previously updated counter to 0 
    clearBeforeNextDilatation(locArr,resShmem,mainQuesCounterArr[!isOddPassShmem[1]+1])
    
    if(blockIdx().x==1)  
        @IfYX 8 8 tailCounter[1]=cld(mainQuesCounterArr[isOddPassShmem[1]+1],(numberOfThreadBlocks+tailParts))*numberOfThreadBlocks +1
    end#if
    loadDataAtTheBegOfDilatationStep(isOddPassShmem,iterationNumberShmem,iterationNumber,positionInMainWorkQueaue,workCounterInshmem,mainQuesCounterArr,isAnyBiggerThanZero,goldToBeDilatated,segmToBeDilatated, resArraysCounters)
    
end    #prepareForNextDilation


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
    clearSourceShmem(sourceShmem)
    clearSharedMemWarpLong(shmemSum, UInt8(14), Float32(0.0))
end#clearBeforeNextDilatation

"""
loads data at the begining of each dilatation step
    we need to set some variables in shared memory ro initial values
"""
function loadDataAtTheBegOfDilatationStep(isOddPassShmem,iterationNumberShmem,iterationNumber,positionInMainWorkQueaue,workCounterInshmem,mainQuesCounterArr,isAnyBiggerThanZero,goldToBeDilatated,segmToBeDilatated, resArraysCounters  )
    #so we know that becouse of sync grid we will have evrywhere the same  iterationNumberShmem and positionInMainWorkQueaue
    
    @IfXY 1 1 iterationNumberShmem[1]+=1
    @IfXY 2 2 positionInMainWorkQueaue[1]=0 

    @IfXY 3 3 begin
        workCounterInshmem[1]= mainQuesCounterArr[isOddPassShmem[1]+1]
        workCounterBiggerThan0[1]= (workCounterInshmem[1]>0)
                    end 
    @IfXY 4 4 goldToBeDilatated[1]=(resArraysCounters[2] < fp[1])
    @IfXY 5 5 segmToBeDilatated[1]=(resArraysCounters[1] < fn[1])

end
