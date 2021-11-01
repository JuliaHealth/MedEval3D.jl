module MainLoopKernel
using CUDA, Logging,Main.CUDAGpuUtils, Main.ResultListUtils,Main.WorkQueueUtils,Main.ScanForDuplicates, Logging,StaticArrays, Main.IterationUtils, Main.ReductionUtils, Main.CUDAAtomicUtils,Main.MetaDataUtils
using Main.MetadataAnalyzePass, Main.ScanForDuplicates
export @iterateOverWorkQueue,@clearBeforeNextDilatation,@mainLoop,@mainLoopKernelAllocations,@innersingleDataBlockPass,@privateWorkQueueAnalysis, @analyzeTail,@prepareForNextDilation
export mainLoopKernel




"""
clearing data in order to enable their reuse in next iteration
clearIterResShmemLoop - given we trat res shmem as one dimensional array with 32 entries per iteration how many times we need to loop to cover all
clearIterSourceShmemLoop - the same as above but for source shmem
resShmemTotalLength, sourceShmemTotalLength - total (treating as 1 dimensional array) length of the  sourse or result shared memory
"""
macro clearBeforeNextDilatation( clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength, sourceShmemTotalLength)
    return esc(quote
    #register variables
    # locArr=0
    # offsetIter=0
    # localOffset=0
    # @ifXY 1 9 oldBlockCounter[1]=0
    #resetting the counter so when we will add to this new items it will count from 0 
    @iterateLinearly clearIterResShmemLoop resShmemTotalLength  resShmem[i]=false
    @iterateLinearly clearIterSourceShmemLoop sourceShmemTotalLength  sourceShmem[i]=false
    
    # TODO () is it needed in dilatations or only in metadata pass
    # for i in 1:14
    #     @exOnWarp i shmemSum[threadIdxX(), i]
    # end     
    # !!!!!! clear alreadyCoveredInQueues
end)#quote
end#clearBeforeNextDilatation



"""
allocates memory in the kernel some register and shared memory  (no allocations in global memory here from kernel)
datBdim - are dimensions of data block - each data block has just one row in the metadata ...

"""
macro mainLoopKernelAllocations(datBdim)
    return esc(quote
    #needed to manage cooperative groups functions
 grid_handle = this_grid()
 #storing intermidiate results +2 in order to get the one padding 
 resShmem =  @cuDynamicSharedMem(Bool,($datBdim[1]+2,$datBdim[2]+2,$datBdim[3]+2)) # we need this additional 33th an 34th spots
 #storing values loaded from analyzed array ...
 sourceShmem =  @cuDynamicSharedMem(Bool,($datBdim[1],$datBdim[2],$datBdim[3]))
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
 localOffset= UInt32(0)
 #we will store here the indexes about the blocks that we want to process 
 toIterWorkQueueShmem =  @cuStaticSharedMem(UInt32,32)
 #indicates where we are in general work queue in given moment if we are iterating over part of work queue owned by this thread block
 positionInLocalWorkQueaue =  @cuStaticSharedMem(UInt16,1)
 #boolean usefull in the iterating over private part of work queue 
 isAnyBiggerThanZero =  @cuStaticSharedMem(Bool,1)
#  #used for iterating over a tail
#  tailCounterInShmem= @cuStaticSharedMem(UInt16,1)
#  workCounterInshmem= @cuStaticSharedMem(UInt16,1)

 #loading data to shared memory from global about is it even or odd pass
 isOddPassShmem =  @cuStaticSharedMem(Bool,1)
 iterationNumberShmem =  @cuStaticSharedMem(UInt16,1)
 #current spot in tail - usefull to save where we need to access the tail to get proper block
#  currentTailPosition =  @cuStaticSharedMem(UInt16,1)
 #shared memory variables needed to marks wheather we are already finished with  any dilatation step
 goldToBeDilatated =  @cuStaticSharedMem(Bool,1)
 segmToBeDilatated =  @cuStaticSharedMem(Bool,1)
 #true when we have more than 0 blocks to analyze in next iteration
 workCounterBiggerThan0 =  @cuStaticSharedMem(Bool,1) 
 @ifXY 1 1 iterationNumberShmem[1]= 0
 @ifXY 2 1 isAnyBiggerThanZero[1]= 0
#  @ifXY 3 1 tailCounterInShmem[1]= 0
 @ifXY 4 1 workCounterInshmem[1]= 0
 @ifXY 5 1 isOddPassShmem[1]= 0
 @ifXY 6 1 currentTailPosition[1]= 0
 @ifXY 7 1 goldToBeDilatated[1]= 0
 @ifXY 8 1 segmToBeDilatated[1]= 0
 @ifXY 9 1 workCounterBiggerThan0[1]= 0
    
 @clearBeforeNextDilatation(clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength, sourceShmemTotalLength)
 sync_threads()
end)#quote
end    


"""
stores the most important part of the kernel where we analyze supplied data blocks
do the dilatation and add to the result queue
"""
macro  innersingleDataBlockPass()
   return esc(quote

            #in order to be able to skip some of the validations we will load now informations about this block and neighbouring blocks 
           #like for example are there any futher results to be written in this block including border queues
           #and is there sth in border queues of the neighbouring data blocks

           sync_threads()
           ############### execution
           #@executeDataIterWithPadding() 
end) #quote
end





"""
main loop logic
"""
macro mainLoop()
    return esc(quote
    MetadataAnalyzePass.@setMEtaDataOtherPasses(locArr,offsetIter,iterThrougWarNumb, globalCurrentFpCount, globalCurrentFnCount)
    # loadOwnedWorkQueueIntoShmem(mainWorkQueue,mainQuesCounter,toIterWorkQueueShmem,positionInMainWorkQueaue ,numberOfThreadBlocks,tailParts)
    # #at this point if we have anything in the private part of the  work queue we will have it in toIterWorkQueueShmem, in case  we will find some 0 inside it means that queue is exhousted and we need to loo into tail
    # #we are analyzing here data  from private part of work queue
    # while(isAnyBiggerThanZero[]) #isAnyBiggerThanZero will be modified inside @privateWorkQueueAnalysis
    #     @privateWorkQueueAnalysis()
    # end#while
    # @ifXY 4 4 currentTailPosition[1]= CUDA.atomic_inc!(pointer(tailCounter), UInt16(1))+1

    # # in this moment we have nothing left in private part of work queue and we need to check is there sth in tail to process
    # #first load data of tail counter once

    # sync_threads()
    # while (currentTailPosition[1]< workCounterInshmem[1])
    #     @analyzeTail()
    # end

    # #we are just negiting it so it should be the same in all blocks
    # @ifXY 1 1 isOddPassShmem[1]= !isOddPassShmem[1]
    # #if we are here it means we had covered all blocks that were marked as active and we need to prepare to next dilatation step

    # sync_grid(grid_handle)
    #     @prepareForNextDilation()
    # sync_grid(grid_handle)
end) #quote
end#mainLoop


"""
iterating over elements in work queaue  in order to make it work we need  to 
    know how many elements are in the work queue - we need workQueauecounter
    workQueaue - matrix with work queue data  
    goldToBeDilatated, segmToBeDilatated - shared memory values needed to establish weather we finished dilatations in given pass
    shmemSum will be used to store loaded data from workqueue 
    shmemSumLengthMaxDiv4 - number indicating linear length of shmemSum but reduced such that it will be divisible by 4
    ex - actions invoked on the data block when its xMeta,yMeta,zMeta and is gold pass informations are already known
    """
macro iterateOverWorkQueue(workQueauecounter,workQueaue,goldToBeDilatated, segmToBeDilatated,shmemSumLengthMaxDiv4,ex )
   return esc(quote
    #first part we load data from work queue to shmem sum 
    # we will treat shmemSum as 1 dimensional array and write data from work queue
    #mod 1 - xMeta, mod 2 - uMeta, mod 3 - zMeta mod 4 - isGoldPass
    #first we need to  establish how many items in work queue will be analyzed by this block 
    numbOfDataBlockPerThreadBlock = cld(workQueauecounter[1],gridDimX() )

    #we need to stuck all of the blocks data into shared memory 4 entries for each block
    @unroll for outerIter in 0: fld((numbOfDataBlockPerThreadBlock),shmemSumLengthMaxDiv4)
        workQuueueLinearIndexOffset = (((numbOfDataBlockPerThreadBlock-1)*4)*(blockIdxX()-1))+(outerIter*shmemSumLengthMaxDiv4)
        # now we load all needed data into shared memory
        @iterateLinearly cld(blockDimX()*blockDimY(),shmemSumLengthMaxDiv4) shmemSumLengthMaxDiv4 begin
            #checking if we are in range
            workQuueueLinearIndex =workQuueueLinearIndexOffset +i
            if(workQuueueLinearIndex<=(workQueauecounter[1]*4))
                #CUDA.@cuprint "workQuueueLinearIndex $(Int64(workQuueueLinearIndex))  (((numbOfDataBlockPerThreadBlock-1)*4) $((((numbOfDataBlockPerThreadBlock-1)*4)))  (((numbOfDataBlockPerThreadBlock-1)*4)*(blockDimX()-1)) $((((numbOfDataBlockPerThreadBlock-1)*4)*(blockDimX()-1))) (outerIter*shmemSumLengthMaxDiv4) $((outerIter*shmemSumLengthMaxDiv4))  workQueaue[workQuueueLinearIndex] $(Int64(workQueaue[workQuueueLinearIndex])) \n"
                shmemSum[i]= workQueaue[workQuueueLinearIndex]  
            end  
        end
    #at this point we had pushed into shared memory as much data as we can fit so we need to start using it         
    #second part proper iteration by definition no rounding needed here 
    #also we do not make here any attempt of parallelization as this will be done inside the expression we just provide actual metadata for dilatation step
        for shmemIndex in 0:fld(shmemSumLengthMaxDiv4,4)
            if((workQuueueLinearIndexOffset+shmemIndex)<numbOfDataBlockPerThreadBlock )

            #data used block wide
            xMeta= shmemSum[shmemIndex*4+1]
            yMeta= shmemSum[shmemIndex*4+2]
            zMeta= shmemSum[shmemIndex*4+3]
            isGold= shmemSum[shmemIndex*4+4]
            #checking is there any point in futher dilatations of this block
            if((isGold==1 && goldToBeDilatated[1]) || (isGold==0 && segmToBeDilatated[1]) )
                #finally all ready for dilatation step to be executed on this particular block
                @ifXY 1 1  CUDA.@cuprint " xMeta $(xMeta) yMeta $(yMeta)  zMeta $(zMeta) isGold $(isGold) \n "
                $ex 
            end    
            end#if in range
        end# main functional loop for dilatation and validation  
    end#outer for 
    end)#quote 
end    

"""
main kernel managing 
"""
function mainLoopKernel()

    @mainLoopKernelAllocations(datBdim)
    MetadataAnalyzePass.@analyzeMetadataFirstPass()
    @ifXY 1 1 iterationNumberShmem[1]+=1
    sync_grid(grid_handle)
    #loadDataAtTheBegOfDilatationStep(true,iterationNumberShmem,iterationNumber,positionInMainWorkQueaue,workCounterInshmem,mainQuesCounterArr,isAnyBiggerThanZero,goldToBeDilatated,segmToBeDilatated, resArraysCounters  )
    
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


# """
# analyzing private part of work queue
# """
# macro privateWorkQueueAnalysis()
#     return esc(quote
 
#     loadOwnedWorkQueueIntoShmem(mainWorkQueue,mainQuesCounter,toIterWorkQueueShmem,positionInMainWorkQueaue ,numberOfThreadBlocks,tailParts)
#     #at this point if we have anything in the private part of the  work queue we will have it in toIterWorkQueueShmem, in case  we will find some 0 inside it means that queue is exhousted and we need to loo into tail
#     @unroll for i in UInt16(1):32# most outer loop is responsible for z dimension
#         if(toIterWorkQueueShmem[i,1]>0)
#             #we need to check also wheather given dilatation step is already finished - for example it is possible that it do not make sense to dilatate gold mask but still we need dilate other
#             if( (goldToBeDilatated[1]&&toIterWorkQueueShmem[i,4]==1) || (segmToBeDilatated[1]&&toIterWorkQueueShmem[i,4]==0)  )    
#                 @innersingleDataBlockPass(toIterWorkQueueShmem[i,4]==1,toIterWorkQueueShmem[i,1],toIterWorkQueueShmem[i,2] ,toIterWorkQueueShmem[i,3] )
#             end
#             sync_threads()
#         else    #at this point we got some 0 in the toIterWorkQueueShmem 
#             @ifXY 1 1  isAnyBiggerThanZero[]=false
#             sync_threads()
#         end#if    
#     end#for    
# end)#quote
# end    


# """
# analyzing tail part of work queue
# """
# macro analyzeTail()
#     return esc(quote

#     #below we access tail of working queue in a way that will be atomic
#     @ifXY 1 1 toIterWorkQueueShmem[1]= mainWorkQueueArr[isOddPassShmem[]+1,currentTailPosition[1]]
#     sync_threads()
#     #we need to check also wheather given dilatation step is already finished - for example it is possible that it do not make sense to dilatate gold mask but still we need dilate other
#     if( (goldToBeDilatated[1]&&toIterWorkQueueShmem[1,4]==1) || (segmToBeDilatated[1]&&toIterWorkQueueShmem[1,4]==0)  )    
#         @innersingleDataBlockPass(toIterWorkQueueShmem[1,4]==1,toIterWorkQueueShmem[1,1],toIterWorkQueueShmem[1,2] ,toIterWorkQueueShmem[1,3] )
#     end
#     loadDataNeededForTailAnalysisToShmem(currentTailPosition,tailCounter )
#     sync_threads()
# end) #quote
# end




"""
after dilatation prepare
"""
macro prepareForNextDilation()
    return esc(quote
    #we update metadata and prepare work queue
    setMEtaDataOtherPasses()
    #we clear  and add negation to !isOddPassShmem becouse we want to set the previously updated counter to 0 
    @clearBeforeNextDilatation( clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength, sourceShmemTotalLength)

    loadDataAtTheBegOfDilatationStep(isOddPassShmem,iterationNumberShmem,iterationNumber,positionInMainWorkQueaue,workCounterInshmem,mainQuesCounterArr,isAnyBiggerThanZero,goldToBeDilatated,segmToBeDilatated, resArraysCounters)
end)#quote
end    #prepareForNextDilation



"""
loads data at the begining of each dilatation step
    we need to set some variables in shared memory ro initial values
"""
function loadDataAtTheBegOfDilatationStep(isOddPassShmem,iterationNumberShmem,iterationNumber,positionInMainWorkQueaue,workCounterInshmem,mainQuesCounterArr,isAnyBiggerThanZero,goldToBeDilatated,segmToBeDilatated, resArraysCounters  )
    #so we know that becouse of sync grid we will have evrywhere the same  iterationNumberShmem and positionInMainWorkQueaue
    
    @ifXY 1 1 iterationNumberShmem[1]+=1
    #@ifXY 2 2 positionInMainWorkQueaue[1]=0 

    @ifXY 3 3 begin
        workCounterInshmem[1]= mainQuesCounterArr[isOddPassShmem[1]+1]
        workCounterBiggerThan0[1]= (workCounterInshmem[1]>0)
                    end 
    @ifXY 4 4 goldToBeDilatated[1]=(resArraysCounters[2] < fp[1])
    @ifXY 5 5 segmToBeDilatated[1]=(resArraysCounters[1] < fn[1])

end


end#MainLoopKernel