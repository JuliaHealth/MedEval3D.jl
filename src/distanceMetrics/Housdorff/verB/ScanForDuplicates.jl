module ScanForDuplicates
using CUDA, Logging,Main.CUDAGpuUtils,Main.WorkQueueUtils,Main.ScanForDuplicates, Logging,StaticArrays, Main.IterationUtils, Main.ReductionUtils, Main.CUDAAtomicUtils,Main.MetaDataUtils
export @loadAndScanForDuplicates, scanForDuplicatesB,scanForDuplicatesMainPart,scanWhenDataInShmem,manageDuplicatedValue




"""
after previous sync threads we already have the number of how much we increased number of results  relative to previous dilatation step
now we need to go through  those numbers and in case some of the border queues were incremented we need to analyze those added entries to establish is there 
any duplicate in case there will be we need to decrement counter and set the corresponding duplicated entry to 0 
"""
function scanForDuplicatesB(oldCount, countDiff,innerWarpNumb,resShmem,shmemSum,resListIndicies,metaData,xMeta,yMeta,zMeta,metaDataDims,localOffset,maxResListIndex,outerWarpLoop) 
    #<13 becouse we are intrested only in border queues as only in those we can have overlap
    if(innerWarpNumb<13)
        @exOnWarp innerWarpNumb begin
           #we loaded data into threades registers so now we need to iteratively go through this and add this register info to shared memory to be available to all threads 
           @unroll for threadNumber in 1:32 # we need to analyze all thread id x 
                if( resShmem[(threadIdxX())+(innerWarpNumb)*33]) # futher actions necessary only if counter diffrence is bigger than 0 
                    if( threadIdxX()==threadNumber ) #now we need some  values that are in the registers  of the associated thread 
                        #those will be needed to know what we need to iterate over 
                        #basically we will start scanning from queaue offset + old count untill queue offset + new count
                        shmemSum[33,innerWarpNumb]=  oldCount 
                        shmemSum[34,innerWarpNumb]=  countDiff
                        # #queue result offset
                        shmemSum[35,innerWarpNumb]= localOffset
                        #will be used in order to keep track of proper new size of counter - will stay 0 if we have no duplicates
                        shmemSum[36,innerWarpNumb]= 0
                    end# if ( threadIdxX()==warpNumb )
                end# if resShmem[(threadIdxX())+(innerWarpNumb)*33]
                sync_warp()# now we should have the required number for scanning of new values for duplicates important that we are analyzing now given queue with just single warp                  
                    scanForDuplicatesMainPart(shmemSum,innerWarpNumb,resListIndicies,metaData,xMeta,yMeta,zMeta,resShmem,metaDataDims,threadNumber,maxResListIndex,outerWarpLoop)
                sync_warp()
            end # for warp number  
        end #exOnWarp
    end    
end

"""
main part of scanninf we are already on a correct warp; we already have old counter value and new counter value available in shared memory
now we need to access the result queue starting from old counter 
"""
function  scanForDuplicatesMainPart(shmemSum,innerWarpNumb,resListIndicies,metaData,xMeta,yMeta,zMeta,resShmem,metaDataDims,threadNumber,maxResListIndex,outerWarpLoop)
    #this hold information wheather we have any new entries in a result queue
    if(shmemSum[34,innerWarpNumb]>0 )
        #as we can analyze 32 numbers at once if the amount of new results is bigger we need to do it in multiple passes 
        @unroll for scanIter in 0:fld(shmemSum[34,innerWarpNumb],32)
                # here we are loading data about linearized indicies of result in main  array depending on a queue we are analyzing it will tell about gold or other pas
                if(((scanIter*32) + threadIdxX())< shmemSum[34,innerWarpNumb]  )
                     indexx = (shmemSum[33,innerWarpNumb]  +shmemSum[35,innerWarpNumb] + (scanIter*32) + threadIdxX())
                    shmemSum[threadIdxX(),innerWarpNumb] = resListIndicies[indexx]
                end
            sync_warp() # now we have 32 linear indicies loaded into the shared memory
            #so we need to load some value into single value into thread and than go over all value in shared memory  
            scanWhenDataInShmem(shmemSum,innerWarpNumb, scanIter,resListIndicies,metaData,xMeta,yMeta,zMeta,metaDataDims,threadNumber,maxResListIndex,outerWarpLoop  )           
            sync_warp()
        end# for scanIter 

            #if this below it means that we have removed some entries from result list; we also check for thread to get appropriate metadata location
            if( (shmemSum[36,innerWarpNumb]>0 ) && (threadIdxX()==threadNumber)  )

                metaData[xMeta,yMeta+1,zMeta+1,((getNewCountersBeg()) +innerWarpNumb) ]= (shmemSum[33,innerWarpNumb]+shmemSum[34,innerWarpNumb])-shmemSum[36,innerWarpNumb]
              #resetting the shmem values  that potentially could be not overwritten
                  shmemSum[34,innerWarpNumb]= 0
                  shmemSum[35,innerWarpNumb]=0
                  shmemSum[36,innerWarpNumb]=0
             end   
             sync_warp()

    end#if    
end


"""
in this spot we already have 32 (not more at least ) values in shared memory
we need now only to establish are there any duplicates
"""
function scanWhenDataInShmem(shmemSum,innerWarpNumb, scanIter,resListIndicies,metaData,xMeta,yMeta,zMeta,metaDataDims,threadNumber,maxResListIndex ,outerWarpLoop )
    @unroll for tempCount in (shmemSum[33,innerWarpNumb] ):(shmemSum[33,innerWarpNumb]+shmemSum[34,innerWarpNumb])
        resListCurrIndex = tempCount +shmemSum[35,innerWarpNumb]
        #now we need to make sure that we are not at spot whre this value is legitimite - so this is first occurence
        if((resListCurrIndex!=( (shmemSum[33,innerWarpNumb]  +shmemSum[35,innerWarpNumb] + (scanIter*32) + threadIdxX()) ) ) && resListCurrIndex < maxResListIndex  )
            #finally we iterate over all values in any given thread and compare to associated value in shared memory
            if(shmemSum[threadIdxX(),innerWarpNumb]>0)
               #we need also to remember that we can have only 2 copies of the same result entry  we will keep only first one and second one we will remove
                if( (resListIndicies[resListCurrIndex]  == shmemSum[threadIdxX(),innerWarpNumb]) &&  (tempCount>  ((scanIter*32) + threadIdxX()))   )
                    #incrementing shared memory to later acrualize counter
                    @atomic shmemSum[36,innerWarpNumb]+=1
                    #if we have repeated value one entry in ids we set to 0
                    resListIndicies[resListCurrIndex]=0
                end
            end    
        end    
    end     
end #scanWhenDataInShmem





    """
    here we will mark in metadata weather there is anything to be verified - here in given que ie - weather it is possible in given queue to cover anything more in next dilatation step
    so it is important for analysisof this particular block is  there is true - is there is non 0 amount of points to cover in any queue of the block
    simultaneously the border queues should indicate for neighbouring blocks (given they exist ) is there is any point in analyzing the paddings ...
    """
    function setIsToBeValidated(innerWarpNumb,offsetIter,metaData,xMeta,yMeta,zMeta)
      # @exOnWarp (innerWarpNumb +14) metaData[xMeta,yMeta+1,zMeta+1,getIsToBeNotAnalyzedNumb()+innerWarpNumb ] = (offsetIter==0) 
    
    end#setIsToBeValidated
    





    """
    as we are operating under assumption that we do not know how many warps we have - we do not know the y dimension of thread block we need to load data into registers with a loop 
    and within the same loop scan it for duplicates
    so if we have more than 12 warps we will execute the loop once - in case we have more we need to execute it in the loop
    iterThrougWarNumb - indicates how many times we need to  iterate to cover all 12 ques if we have at least 12 warps available (ussually we will) we will execute it once
    locArr,offsetIter,localOffset - variables used for storing some important constants
    
    """
    macro loadAndScanForDuplicates(iterThrougWarNumb,locArr,offsetIter,localOffset)

        return esc(quote

        @unroll for outerWarpLoop in 0:$iterThrougWarNumb     
                #represents the number of queue if we have enought warps at disposal it equals warp number so idY
                innerWarpNumb = (threadIdxY()+ outerWarpLoop*blockDimY())
                #now we will load the diffrence between old and current counter if we are in range of metadata
                if(( innerWarpNumb)<15 && isInRange)
                    @exOnWarp innerWarpNumb begin
                        #store result in registers
                        #old count
                        $locArr =  metaData[xMeta,yMeta+1,zMeta+1,((getOldCountersBeg()) +innerWarpNumb) ]
                        #diffrence new - old 
                        $offsetIter=  metaData[xMeta,yMeta+1,zMeta+1,((getNewCountersBeg()) +innerWarpNumb) ] - $locArr

                        $localOffset = metaData[xMeta,yMeta+1,zMeta+1, getResOffsetsBeg()+innerWarpNumb]
                        #### here we will mark in metadata weather there is anything to be verified - here in given que ie - weather it is possible in given queue to cover anything more in next dilatation step
                        #setIsToBeValidated(innerWarpNumb,offsetIter,metaData,xMeta,yMeta,zMeta) 
                        # enable access to information is it bigger than 0 to all threads in block
                        resShmem[(threadIdxX())+(innerWarpNumb)*33] = $offsetIter >0
                    end #@exOnWarp
                end#if
                sync_threads()
                #main function for scanning
                scanForDuplicatesB($locArr, $offsetIter,innerWarpNumb,resShmem,shmemSum,resListIndicies,metaData,xMeta,yMeta,zMeta,metaDataDims,localOffset,maxResListIndex,outerWarpLoop) 
            end#for  
        end)#quote
             
                    
        end #loadAndScanForDuplicates    
    




end#module





# """
# this will be invoked when we have duplicated value in result queue - so we need to 
#     set this value - of linear index to 0 
# """
# function  manageDuplicatedValue(resListIndicies,tempCount, metaData,xMeta,yMeta,zMeta ,innerWarpNumb ,metaDataDims)
#     # resList[tempCount,1]=0
#     # resList[tempCount,2]=0
#     # resList[tempCount,3]=0
#     # resList[tempCount,4]=0
#     # resList[tempCount,5]=0
#     # resList[tempCount,6]=0

#     # CUDA.@cuprint "manageDuplicatedValue  tempCount $(tempCount)  xMeta $(xMeta) yMeta $(yMeta) zMeta $(zMeta) innerWarpNumb $(innerWarpNumb)    \n"
#     resListIndicies[tempCount]=0
#     #indd = (xMeta+ (yMeta)*metaDataDims[1]+(zMeta)*metaDataDims[1]*metaDataDims[2]+((getNewCountersBeg()) +innerWarpNumb -1) *metaDataDims[1]*metaDataDims[2] *metaDataDims[3])
#     # if(indd==1102501)
#     #     CUDA.@cuprint " indd $(indd)  xMeta $(xMeta) yMeta $(yMeta+1) zMeta $(zMeta+1) innerWarpNumb $(innerWarpNumb) tempCount $(tempCount) \n"
#     # end
#     #pointer(metaData,indd)
#    #@atomic 
#    #metaData[indd]-=1 #metaData[indd]-UInt32(1)    
# end