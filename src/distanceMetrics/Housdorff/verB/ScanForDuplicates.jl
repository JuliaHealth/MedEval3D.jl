module ScanForDuplicates
using CUDA, Logging,..CUDAGpuUtils,..WorkQueueUtils,..ScanForDuplicates, Logging,StaticArrays, ..IterationUtils, ..ReductionUtils, ..CUDAAtomicUtils,..MetaDataUtils
export @getIsToVal,@loadAndScanForDuplicates,@setIsToBeValidated, @scanForDuplicatesB,scanForDuplicatesMainPart,scanWhenDataInShmem,manageDuplicatedValue




# """
# after previous sync threads we already have the number of how much we increased number of results  relative to previous dilatation step
# now we need to go through  those numbers and in case some of the border queues were incremented we need to analyze those added entries to establish is there 
# any duplicate in case there will be we need to decrement counter and set the corresponding duplicated entry to 0 
# """
# macro scanForDuplicatesB(oldCount, countDiff,localOffset)#innerWarpNumb,shmemblockData,shmemSum,resListIndicies,metaData,xMeta,yMeta,zMeta,metaDataDims,localOffset,maxResListIndex,outerWarpLoop,alreadyCoveredInQueues,sourceShmem) 
#     return esc(quote
#     #<13 becouse we are intrested only in border queues as only in those we can have overlap
#     if(innerWarpNumb<13)
#         sync_threads()
#         @exOnWarp innerWarpNumb begin 
#            #we loaded data into threades registers so now we need to iteratively go through this and add this register info to shared memory to be available to all threads 
#            @unroll for threadNumber in 1:32 # we need to analyze all thread id x 
#                     if( threadIdxX()==threadNumber ) #now we need some  values that are in the registers  of the associated thread 
#                         shmemSum[33,innerWarpNumb]=  $oldCount 
#                         shmemSum[34,innerWarpNumb]=  $countDiff
#                         # #queue result offset
#                         shmemSum[35,innerWarpNumb]= $localOffset
#                         #will be used in order to keep track of proper new size of counter - will stay 0 if we have no duplicates
#                         shmemSum[36,innerWarpNumb]= 0
#                     end# if ( threadIdxX()==warpNumb )
#                 sync_warp()# now we should have the required number for scanning of new values for duplicates important that we are analyzing now given queue with just single warp - so we also get separate blocks as each block is represented by given idX              
#                     scanForDuplicatesMainPart(shmemSum,innerWarpNumb,resListIndicies,metaData,xMeta,yMeta,zMeta,shmemblockData,metaDataDims,threadNumber,maxResListIndex,outerWarpLoop,alreadyCoveredInQueues, isInRange,resList,dataBdim)
#                 sync_threads()
#             end # for threadNumber 
#         end #exOnWarp
#     end
# end)#qote       
# end

# """
# main part of scanninf we are already on a correct warp; we already have old counter value and new counter value available in shared memory
# now we need to access the result queue starting from old counter 
# """
# function  scanForDuplicatesMainPart(shmemSum,innerWarpNumb,resListIndicies,metaData,xMeta,yMeta,zMeta,shmemblockData,metaDataDims,threadNumber,maxResListIndex,outerWarpLoop,alreadyCoveredInQueues, isInRange,resList,dataBdim)
#     #this hold information wheather we have any new entries in a result queue
#     if(shmemSum[34,innerWarpNumb]>0 )
#         #as we can analyze 32 numbers at once if the amount of new results is bigger we need to do it in multiple passes 
#         @unroll for scanIter in 0:fld(shmemSum[34,innerWarpNumb],32)
#                 # here we are loading data about linearized indicies of result in main  array depending on a queue we are analyzing it will tell about gold or other pas
#                     entryIndex = (shmemSum[33,innerWarpNumb]  +shmemSum[35,innerWarpNumb] + (scanIter*32) + threadIdxX())
#                     shmemSum[threadIdxX(),innerWarpNumb] = resListIndicies[entryIndex]
#                     #now becouse the entries from  paddings had not been saved into the dilatation arrays we need to do it now 
#                     if(resList[7]==1) #checking wheather result entry originated from padding 
#                         #becouse we are inside threadNumber loop we are sure that this datablock will not by modified by any other warp - the only thing we need to be sure is that two threads from the same warp will not try to modify the same integer ...
#                         #situation is diffrent for diffrent queues 
#                             #in case of top and down we do not need to be afreaid of overwriting
#                             if(innerWarpNumb>8 && innerWarpNumb<13)
                                
#                             end    
#                             #for anterior posterior we need to be sure that xses are diffrent
#                             #for left and right we need to be sure that y are diffrent

#                         krowa

#                     end    
#             sync_warp() # now we have 32 linear indicies loaded into the shared memory
#             #so we need to load some value into single value into thread and than go over all value in shared memory  
#             scanWhenDataInShmem(shmemSum,innerWarpNumb, scanIter,resListIndicies,metaData,xMeta,yMeta,zMeta,metaDataDims,threadNumber,maxResListIndex,outerWarpLoop ,alreadyCoveredInQueues )           
#             sync_warp()
#         end# for scanIter 
#             ## all of those below happen only on one lane of each warp -  distributing work we nned to look for ideas to distribute work
#             #if this below it means that we have removed some entries from result list; we also check for thread to get appropriate metadata location
#             if(threadIdxX()==threadNumber)
#                 # the counbter reduced if needed of dupl;cated values
#                 updated = (shmemSum[33,innerWarpNumb]+shmemSum[34,innerWarpNumb])-shmemSum[36,innerWarpNumb]
#                 #alreadyCoveredInQueues will be later used to get global sums
#                 #added in this step 
#                 if((shmemSum[34,innerWarpNumb]-shmemSum[36,innerWarpNumb]) >0)
#                     alreadyCoveredInQueues[innerWarpNumb]+=(shmemSum[34,innerWarpNumb]-shmemSum[36,innerWarpNumb])
 
#                 end    
#                 #we set the counter to new value only if there were some subtractions done - some values were duplicated
#                 if((shmemSum[36,innerWarpNumb]>0 ) && isInRange )
#                     @setMeta(((getNewCountersBeg()) +innerWarpNumb) , updated)
#                   #krr metaData[xMeta+1,yMeta+1,zMeta+1,((getNewCountersBeg()) +innerWarpNumb) ]= updated

#                 end
#                 #it shoud NOT be analyzed if corrected counter value is the same as the total number for this queue
#                 # but we need to analyze this only if there was any new value added relatively to last pass
#                 if( (shmemSum[34,innerWarpNumb]-shmemSum[36,innerWarpNumb])  >0)
#                     # krr if(isInRange && updated!= metaData[(xMeta +1),(yMeta+1),(zMeta+1),getBeginingOfFpFNcounts()+innerWarpNumb ])

#                     if(isInRange && updated!= @accMeta((getBeginingOfFpFNcounts()+innerWarpNumb)) )
#                         #so in res shmem we have information weather we should  validate this queue ...
#                         shmemblockData[(threadIdxX())+(innerWarpNumb+21)*33]= 1
#                     end
                    
#                  end    
#             end   
#              sync_warp()
#              #resetting the shmem values  that potentially could be not overwritten
#             for i in 1:3
#                 @ifX i if(shmemSum[36,innerWarpNumb]>0 )   shmemSum[33+i,innerWarpNumb]=0 end
#             end    
       


#     end#if    
# end


# """
# in this spot we already have 32 (not more at least ) values in shared memory
# we need now only to establish are there any duplicates
# """
# function scanWhenDataInShmem(shmemSum,innerWarpNumb, scanIter,resListIndicies,metaData,xMeta,yMeta,zMeta,metaDataDims,threadNumber,maxResListIndex ,outerWarpLoop ,alreadyCoveredInQueues)
#     @unroll for tempCount in (shmemSum[33,innerWarpNumb]):((shmemSum[33,innerWarpNumb]+shmemSum[34,innerWarpNumb])+1)
#         resListCurrIndex = tempCount +shmemSum[35,innerWarpNumb]
#        #now we need to make sure that we are not at spot whre this value is legitimite - so this is first occurence
#         if((resListCurrIndex!=( (shmemSum[33,innerWarpNumb]  +shmemSum[35,innerWarpNumb] + (scanIter*32) + threadIdxX()) ) ) && resListCurrIndex < maxResListIndex  )
#             #finally we iterate over all values in any given thread and compare to associated value in shared memory
#             if(shmemSum[threadIdxX(),innerWarpNumb]>0)
#                 #now we can make small optimazation and reduce global memory use and if reslist currindex is in range of currently loaded value in shared memory we will load them from shared memory
#                 shmemIndexBase = (shmemSum[33,innerWarpNumb]  +shmemSum[35,innerWarpNumb] + (scanIter*32))
#                 scannedVal = if(resListCurrIndex>shmemIndexBase && resListCurrIndex<(shmemIndexBase+32)  ) # range that is in the shmem
#                                 shmemSum[(resListCurrIndex- shmemIndexBase),innerWarpNumb]
#                             else 
#                                 resListIndicies[resListCurrIndex] 
#                             end
                            
#                 #we need also to remember that we can have only 2 copies of the same result entry  we will keep only first one and second one we will remove
#                 if( ( scannedVal == shmemSum[threadIdxX(),innerWarpNumb]) &&  (tempCount>  ((scanIter*32) + threadIdxX()))   )
#                     #incrementing shared memory to later actualize counter
#                    CUDA.@atomic shmemSum[36,innerWarpNumb]+=1
#                     #if we have repeated value one entry in ids we set to 0
#                     resListIndicies[resListCurrIndex]=0
#                 end
#             end    
#         end    
#     end     
# end #scanWhenDataInShmem




    """
    as we are operating under assumption that we do not know how many warps we have - we do not know the y dimension of thread block we need to load data into registers with a loop 
    and within the same loop scan it for duplicates
    so if we have more than 12 warps we will execute the loop once - in case we have more we need to execute it in the loop
    iterThrougWarNumb - indicates how many times we need to  iterate to cover all 12 ques if we have at least 12 warps available (ussually we will) we will execute it once
    locArr,offsetIter,localOffset - variables used for storing some important constants
    
    """
    macro loadAndScanForDuplicates(iterThrougWarNumb,locArr,offsetIter)

        return esc(quote
            @unroll for outerWarpLoop in 0:$iterThrougWarNumb     
                #represents the number of queue if we have enought warps at disposal it equals warp number so idY
                innerWarpNumb = (threadIdxY()+ outerWarpLoop*blockDimY())
                #at this point we have actual counters with correction for duplicated values  we can compare it with the  total values of fp or fn of a given queue  if we already covered
                #all points of intrest there is no point to futher analyze this block or padding
                
                @setIsToBeValidated()   
                
                if(innerWarpNumb<15)
                    shmemSum[34,innerWarpNumb]= 0
                    shmemSum[35,innerWarpNumb]=0
                    shmemSum[36,innerWarpNumb]=0
                end         
            end#for    
            @exOnWarp 15 if(isInRange) 

            @setMeta((getIsToBeAnalyzedNumb()+15), (@getIsToVal(1) || @getIsToVal(3)|| @getIsToVal(5)|| @getIsToVal(7)|| @getIsToVal(11)|| @getIsToVal(13)) )#sourceShmem[(threadIdxX())+33*8]
            end
            @exOnWarp 16 if(isInRange)

                @setMeta((getIsToBeAnalyzedNumb()+16 ),(@getIsToVal(2) || @getIsToVal(4)|| @getIsToVal(6)|| @getIsToVal(8)|| @getIsToVal(10)|| @getIsToVal(12)) )#sourceShmem[(threadIdxX())+33*6] 
            end
    sync_threads()


end)#quote

             
                    
        end #loadAndScanForDuplicates    
    
        """
        loads value from res shmem about weather a queue with supplied numb has anything worth validating
        """
        macro getIsToVal(numb)
            return esc(quote
            (shmemPaddings[(threadIdxX())+($numb+21)*33])
        end)
        end     

        """
        here we will mark in metadata weather there is anything to be verified - here in given que ie - weather it is possible in given queue to cover anything more in next dilatation step
        so it is important for analysisof this particular block is  there is true - is there is non 0 amount of points to cover in any queue of the block
        simultaneously the border queues should indicate for neighbouring blocks (given they exist ) is there is any point in analyzing the paddings ...
           so we need this data in 3 places 
           1) for the getIsToBeAnalyzedNumb value in metadata of blocks around
           2) for isNotTobeAnalyzed for a current block
           3) also we should accumulate values pf counters - add and reduce across all blocks of tps and fps covered - so we will know when to finish the dilatation steps
     """
       macro setIsToBeValidated()
     return esc(quote
         @exOnWarp innerWarpNumb begin
             if(innerWarpNumb<15 && isInRange)
                #we need also to remember that data wheather there are any futher points of intrest is not only in the current block
                # so here we establish what are the coordinates of metadata of intrest so for example  our left fp and left FN are of intrest to block to the left ...
                newXmeta = xMeta+ (-1 * (innerWarpNumb==1 || innerWarpNumb==2)) + (innerWarpNumb==3 || innerWarpNumb==4)+1
                newYmeta = yMeta+ (-1 * (innerWarpNumb==5 || innerWarpNumb==6)) + (innerWarpNumb==7 || innerWarpNumb==8)+1
                newZmeta = zMeta+ (-1 * (innerWarpNumb==9 || innerWarpNumb==10)) + (innerWarpNumb==11 || innerWarpNumb==12)+1
                # #check are we in range 
                if(newXmeta>0 && newYmeta>0 && newZmeta>0 && newXmeta<=metaDataDims[1] && newYmeta<=metaDataDims[2]  && newZmeta<=metaDataDims[3] && innerWarpNumb<13 )
                    metaData[newXmeta,newYmeta,newZmeta,getIsToBeAnalyzedNumb()+innerWarpNumb ] = (shmemPaddings[(threadIdxX())+(innerWarpNumb+21)*33] )

                end #if in meta data range
             end#if   
             end#ex on warp    
                 
                #here we set the information weather any of the queue related to fp or fn in a particular block  has still something to be analyzed 
     end)#quote
     end#setIsToBeValidated


end#module





# \
# """
# as we are operating under assumption that we do not know how many warps we have - we do not know the y dimension of thread block we need to load data into registers with a loop 
# and within the same loop scan it for duplicates
# so if we have more than 12 warps we will execute the loop once - in case we have more we need to execute it in the loop
# iterThrougWarNumb - indicates how many times we need to  iterate to cover all 12 ques if we have at least 12 warps available (ussually we will) we will execute it once
# locArr,offsetIter,localOffset - variables used for storing some important constants

# """
# macro loadAndScanForDuplicates(iterThrougWarNumb,locArr,offsetIter)

#     return esc(quote

#     # @unroll for outerWarpLoop in 0:$iterThrougWarNumb     
#     #         #represents the number of queue if we have enought warps at disposal it equals warp number so idY
#     #         innerWarpNumb = (threadIdxY()+ outerWarpLoop*blockDimY())
#     #         #now we will load the diffrence between old and current counter if we are in range of metadata
#     #         if(( innerWarpNumb)<15)
#     #             @exOnWarp innerWarpNumb begin
#     #                 if(isInRange)
#     #                     #store result in registers
#     #                     #old count
#     #                     $locArr = @accMeta((getOldCountersBeg()) +innerWarpNumb) 
#     #                     # #diffrence new - old 
#     #                     $offsetIter=   @accMeta(((getNewCountersBeg()) +innerWarpNumb) )- $locArr
#     #                     # #offset to find where are the results associated with given queue
#     #                     $localOffset =  @accMeta((getResOffsetsBeg()+innerWarpNumb))+$locArr
#     #                     # # enable access to information is it bigger than 0 to all threads in block
#     #                     shmemblockData[(threadIdxX())+(innerWarpNumb+6)*33] = UInt32($offsetIter >0)
#     #                     #for all border queues this information will be written down after scanning for duplicates here we are just getting info from main part ...
#     #                 else# if not in range
#     #                     $locArr = 0
#     #                     $offsetIter=0
#     #                     $localOffset=0
#     #                 end#if in range
#     #             end #@exOnWarp
#     #         end#if

#     #         sync_threads()
#     #         # so queaue 13 or 14
#     #             @exOnWarp innerWarpNumb begin
#     #                 if(innerWarpNumb==13 || innerWarpNumb==14)# so queaue 13 or 14
#     #                     if($offsetIter>0) 
#     #                             # krr if($offsetIter!= metaData[xMeta+1,yMeta+1,zMeta+1,getBeginingOfFpFNcounts()+innerWarpNumb ])
#     #                         if($offsetIter!=  @accMeta((getBeginingOfFpFNcounts()+innerWarpNumb)) )
#     #                             shmemblockData[(threadIdxX())+(innerWarpNumb+21)*33]= 1
#     #                             #sourceShmem[(threadIdxX())+33*(6+(isodd(innerWarpNumb) *2) )]= true
#     #                         end
#     #                     end
#     #                     locOffset= UInt16(1)
#     #                     #setting what we need to locArr and reducing value in a warp
#     #                     $locArr+=$offsetIter
#     #                     #TODO (try to do warp reduction below instead of atomic... )
#     #                     if(($offsetIter)>0)
#     #                        CUDA.@atomic alreadyCoveredInQueues[innerWarpNumb]+=($offsetIter)
#     #                     end                       
#     #                     #@redOnlyStepOne(locOffset, shmemSum, $locArr, +)
#     #                     #now we have warp reduced value on first thread
#     #                     # @ifX 1 alreadyCoveredInQueues[innerWarpNumb]+=$locArr
#     #                 end#if    
#     #             end#ex on warp    
#     #             sync_threads()

#     #         #main function for scanning
          
#     #        @scanForDuplicatesB($locArr, $offsetIter,$localOffset)#$locArr, $offsetIter,innerWarpNumb,shmemblockData,shmemSum,resListIndicies,metaData,xMeta,yMeta,zMeta,metaDataDims,$localOffset,maxResListIndex,outerWarpLoop,alreadyCoveredInQueues,sourceShmem)
         
#     #        if(innerWarpNumb<15)
#     #             shmemSum[threadIdxX(),innerWarpNumb]=0
#     #         end


#     #     end#for
#     #     sync_threads()

#         @unroll for outerWarpLoop in 0:$iterThrougWarNumb     
#             #represents the number of queue if we have enought warps at disposal it equals warp number so idY
#             innerWarpNumb = (threadIdxY()+ outerWarpLoop*blockDimY())
#             #at this point we have actual counters with correction for duplicated values  we can compare it with the  total values of fp or fn of a given queue  if we already covered
#             #all points of intrest there is no point to futher analyze this block or padding
            
#             @setIsToBeValidated()   
            
#             if(innerWarpNumb<15)
#                 shmemSum[34,innerWarpNumb]= 0
#                 shmemSum[35,innerWarpNumb]=0
#                 shmemSum[36,innerWarpNumb]=0
#             end         
#         end#for    
#         @exOnWarp 15 if(isInRange) 

#         @setMeta((getIsToBeAnalyzedNumb()+15), (@getIsToVal(1) || @getIsToVal(3)|| @getIsToVal(5)|| @getIsToVal(7)|| @getIsToVal(11)|| @getIsToVal(13)) )#sourceShmem[(threadIdxX())+33*8]
#         end
#         @exOnWarp 16 if(isInRange)

#             @setMeta((getIsToBeAnalyzedNumb()+16 ),(@getIsToVal(2) || @getIsToVal(4)|| @getIsToVal(6)|| @getIsToVal(8)|| @getIsToVal(10)|| @getIsToVal(12)) )#sourceShmem[(threadIdxX())+33*6] 
#         end
# sync_threads()


# end)#quote