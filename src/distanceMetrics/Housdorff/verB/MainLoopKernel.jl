
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
    while(goldToBeDilatated[1] && goldToBeDilatated[1] && workCounterBiggerThan0[1])
        while(isAnyBiggerThanZero[])
            loadOwnedWorkQueueIntoShmem(mainWorkQueue,mainQuesCounter,toIterWorkQueueShmem,positionInMainWorkQueaue ,numberOfThreadBlocks,tailParts)
            #at this point if we have anything in the private part of the  work queue we will have it in toIterWorkQueueShmem, in case  we will find some 0 inside it means that queue is exhousted and we need to loo into tail
            @unroll for i in UInt16(1):32# most outer loop is responsible for z dimension
                if(shmemIter[i,1]>0)
                    #we need to check also wheather given dilatation step is already finished - for example it is possible that it do not make sense to dilatate gold mask but still we need dilate other
                    if( (goldToBeDilatated[1]&&shmemIter[i,4]==1) || (segmToBeDilatated[1]&&shmemIter[i,4]==0)  )    
                        innersingleDataBlockPass(shmemIter[i,4]==1,shmemIter[i,1],shmemIter[i,2] ,shmemIter[i,3] )
                    end
                    sync_threads()
                else    #at this point we got some 0 in the shmemIter 
                    if(threadIdxX()==1)# sadly only threads with this id are managing the work queue
                        isAnyBiggerThanZero[]=false
                    end#inner if    
                end#if    
            end#for    
        end#while
        
        # in this moment we have nothing left in private part of work queue and we need to check is there sth in tail to process
        #first load data of tail counter once
        
        loadDataNeededForTailAnalysisToShmem(currentTailPosition,tailCounter )
        sync_threads()
        while (currentTailPosition[1]< workCounterInshmem[1])
            #below we access tail of working queue in a way that will be atomic
            if(threadIdxY()==1 && threadIdxX()==1 )
                shmemIter[1]= mainWorkQueueArr[isOddPassShmem[]+1][currentTailPosition[1]]
            end
            sync_threads()
            #we need to check also wheather given dilatation step is already finished - for example it is possible that it do not make sense to dilatate gold mask but still we need dilate other
            if( (goldToBeDilatated[1]&&shmemIter[1,4]==1) || (segmToBeDilatated[1]&&shmemIter[1,4]==0)  )    
                innersingleDataBlockPass(shmemIter[1,4]==1,shmemIter[1,1],shmemIter[1,2] ,shmemIter[1,3] )
            end
            loadDataNeededForTailAnalysisToShmem(currentTailPosition,tailCounter )
            sync_threads()
        end
        #we set it before sync grid - as we do not need it for this one
        if(threadIdxX()==10 && threadIdxY()==10)
            #we are just negiting it so it should be the same in all blocks
            isOddPassShmem[1]= !isOddPassShmem[1]
        end

        #if we are here it means we had covered all blocks that were marked as active and we need to prepare to next dilatation step
        sync_grid(grid_handle)
            #we clear  and add negation to !isOddPassShmem becouse we want to set the previously updated counter to 0 
            clearBeforeNextDilatation(locArr,resShmem,mainQuesCounterArr[!isOddPassShmem[1]+1])
            prepareForNextDilatationStep(iterationNumber,tailCounter,numberOfThreadBlocks,tailParts,mainQuesCounterArr,isOddPassShmem)
            loadDataAtTheBegOfDilatationStep(isOddPassShmem,iterationNumberShmem,iterationNumber,positionInMainWorkQueaue,workCounterInshmem,mainQuesCounterArr,isAnyBiggerThanZero,goldToBeDilatated,segmToBeDilatated, resArraysCounters)

        sync_grid(grid_handle)
    end#while isNotFinished    


end

"""
allocates memory in the kernel some register and shared memory  (no allocations in global memory here from kernel)
"""
macro mainLoopKernelAllocations()
 #needed to manage cooperative groups functions
 grid_handle = this_grid()
 #storing intermidiate results +2 in order to get the one padding 
 resShmem =  @cuStaticSharedMem(Bool,(34,34,34))
 #storing values loaded from analyzed array ...
 sourceShmem =  @cuStaticSharedMem(Bool,(34,34,34))
 #for storing sums for reductions
 shmemSum =  @cuStaticSharedMem(Float32,35,14) # we need this additional 33th an 34th spots

 
 #coordinates of data in main array
 #we will use this to establish weather we should mark  the data block as empty or full ...
 isMaskFull= false
 #here we will store in registers data uploaded from mask for later verification wheather we should send it or not
 locArr= Int32(0)
 offsetIter = UInt16(0)

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

"""



"""
loads data at the begining of each dilatation step
    we need to set some variables in shared memory ro initial values
"""
function loadDataAtTheBegOfDilatationStep(isOddPassShmem,iterationNumberShmem,iterationNumber,positionInMainWorkQueaue,workCounterInshmem,mainQuesCounterArr,isAnyBiggerThanZero,goldToBeDilatated,segmToBeDilatated, resArraysCounters  )
    #so we know that becouse of sync grid we will have evrywhere the same  iterationNumberShmem and positionInMainWorkQueaue
    @IfYX 2 2 iterationNumberShmem[1]+=1
    @IfYX 3 3 positionInMainWorkQueaue[1]=0 
    @IfYX 4 4 begin
        workCounterInshmem[1]= mainQuesCounterArr[isOddPassShmem[1]+1]
        workCounterBiggerThan0[1]= (workCounterInshmem[1]>0)
                    end 
    @IfYX 5 5 isAnyBiggerThanZero[]=true  
    @IfYX 6 6 goldToBeDilatated[1]=(resArraysCounters[2] < fp[1])
    @IfYX 7 7 segmToBeDilatated[1]=(resArraysCounters[1] < fn[1])

end
