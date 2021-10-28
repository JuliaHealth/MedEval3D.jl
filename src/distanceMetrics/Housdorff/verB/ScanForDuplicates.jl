module ScanForDuplicates
using CUDA, Logging,Main.CUDAGpuUtils,Main.WorkQueueUtils,Main.ScanForDuplicates, Logging,StaticArrays, Main.IterationUtils, Main.ReductionUtils, Main.CUDAAtomicUtils,Main.MetaDataUtils
export @loadAndScanForDuplicates, @scanForDuplicatesB,@scanForDuplicatesMainPart,@scanWhenDataInShmem,@manageDuplicatedValue




"""
main part of scanninf we are already on a correct warp; we already have old counter value and new counter value available in shared memory
now we need to access the result queue starting from old counter 
"""
macro scanForDuplicatesMainPart()
    return esc(quote
    #as we can analyze 32 numbers at once if the amount of new results is bigger we need to do it in multiple passes 
    @unroll for scanIter in 0: cld(shmemSum[34,innerWarpNumb],32 )
        # # here we are loading data about linearized indicies of result in main  array depending on a queue we are analyzing it will tell about gold or other pas
        # if(((scanIter*32) + threadIdxX())< shmemSum[34,innerWarpNumb]  )
        #     #shmemSum[threadIdxX(),innerWarpNumb] = getResLinIndex(resList[(shmemSum[33,innerWarpNumb]  +shmemSum[35,innerWarpNumb] + (scanIter*32) + threadIdxX()  )],mainArrDims  )
        # end
        sync_warp() # now we have 32 linear indicies loaded into the shared memory
        #so we need to load some value into single value into thread and than go over all value in shared memory  
        #@scanWhenDataInShmem()
    end# for scanIter 
end)
end



"""
in this spot we already have 32 (not more at least ) values in shared memory
we need to now 
"""
macro scanWhenDataInShmem()
    return esc(quote
    @unroll for tempCount in (shmemSum[33,innerWarpNumb]+shmemSum[35,innerWarpNumb] ):(shmemSum[33,innerWarpNumb]+shmemSum[34,innerWarpNumb]+shmemSum[33,innerWarpNumb])
        #now we need to make sure that we are not at spot whre this value is legitimite - so this is first occurence
        if(tempCount!=shmemSum[33,innerWarpNumb]+ (scanIter*32) + threadIdxX()  )
            #finally we iterate over all values in any given thread and compare to associated value in shared memory
            if( getResLinIndex(resList[tempCount])  == shmemSum[threadIdxX(),innerWarpNumb] ,mainArrDims)
                 #if we are here it means that we have duplicated value 
                @manageDuplicatedValue()
            end    
        end    
    end     end )
end #scanWhenDataInShmem


"""
this will be invoked when we have duplicated value in result queue - so we need to 
    set this value - of linear index to 0 
    and reduce the counter value 
"""
macro  manageDuplicatedValue()
    return esc(quote
    resList[tempCount,1]=0
    #decrCounterByOne(numb, metaData,shmemSum[35,resQueueNumb])
    
end )
end



"""
after previous sync threads we already have the number of how much we increased number of results  relative to previous dilatation step
now we need to go through  those numbers and in case some of the border queues were incremented we need to analyze those added entries to establish is there 
any duplicate in case there will be we need to decrement counter and set the corresponding duplicated entry to 0 
"""
macro scanForDuplicatesB(oldCount, countDiff) 
    return esc(quote
    if(innerWarpNumb<13)
        @exOnWarp innerWarpNumb begin
            @unroll for threadNumber in 1:32 # we need to analyze all thread id x 
                if( resShmem[(threadIdxX())+(innerWarpNumb)*33]) # futher actions necessary only if counter diffrence is bigger than 0 
                    if( threadIdxX()==threadNumber ) #now we need some  values that are in the registers  of the associated thread 
                        #those will be needed to know what we need to iterate over 
                        #basically we will start scanning from queaue offset + old count untill queue offset + new count
                        shmemSum[33,innerWarpNumb]=  $oldCount 
                        shmemSum[34,innerWarpNumb]=  $countDiff
                        #shmemSum[35,innerWarpNumb]=  localResOffset
                    end# if ( threadIdxX()==warpNumb )
                end# resShmem[warpNumb+1,i+1,3]
                sync_warp()# now we should have the required number for scanning of new values for duplicates important that we are analyzing now given queue with just single warp
                   @scanForDuplicatesMainPart()
                sync_warp()
            end # for warp number  
        end #exOnWarp
    end    
end )
end


"""
as we are operating under assumption that we do not know how many warps we have - we do not know the y dimension of thread block we need to load data into registers with a loop 
and within the same loop scan it for duplicates
so if we have more than 12 warps we will execute the loop once - in case we have more we need to execute it in the loop
iterThrougWarNumb - indicates how many times we need to  iterate to cover all 12 ques if we have at least 12 warps available (ussually we will) we will execute it once
"""
macro loadAndScanForDuplicates(iterThrougWarNumb,locArr,offsetIter)
        
    return esc(quote

    @unroll for outerWarpLoop in 0:$iterThrougWarNumb     
            innerWarpNumb = (threadIdxY()+ outerWarpLoop*blockDimY())
            #now we will load the diffrence between old and current counter
            if(( innerWarpNumb)<15)
                @exOnWarp innerWarpNumb begin
                    #store result in registers
                    #store result in registers (we are reusing some variables)
                    #old count
                    $locArr =  metaData[xMeta,yMeta+1,zMeta+1,((getOldCountersBeg()-1) +innerWarpNumb) ]
                    #diffrence new - old 
                    $offsetIter=  metaData[xMeta,yMeta+1,zMeta+1,((getNewCountersBeg()-1) +innerWarpNumb) ] - $locArr
                    # #queue result offset
                    # localResOffset = metaData[xMeta,yMeta+1,zMeta+1, getBeginnigOfOffsets()+innerWarpNumb] # tis queue offset
                    
                    #### here we will mark in metadata weather there is anything to be verified - here in given que ie - weather it is possible in given queue to cover anything more in next dilatation step
                    #setIsToBeValidated(innerWarpNumb,offsetIter,metaData,xMeta,yMeta,zMeta)
                    
                    
                    # enable access to information is it bigger than 0 to all threads in block
                    resShmem[(threadIdxX())+(innerWarpNumb)*33] = $offsetIter >0
                end #@ifY
            end#if
            sync_threads()
            @scanForDuplicatesB(locArr, offsetIter) 
        end#for  
    end)#quote
         
                
    end #loadAndScanForDuplicates    



    """
    here we will mark in metadata weather there is anything to be verified - here in given que ie - weather it is possible in given queue to cover anything more in next dilatation step
    so it is important for analysisof this particular block is  there is true - is there is non 0 amount of points to cover in any queue of the block
    simultaneously the border queues should indicate for neighbouring blocks (given they exist ) is there is any point in analyzing the paddings ...
    """
    function setIsToBeValidated(innerWarpNumb,offsetIter,metaData,xMeta,yMeta,zMeta)
      # @exOnWarp (innerWarpNumb +14) metaData[xMeta,yMeta+1,zMeta+1,getIsToBeNotAnalyzedNumb()+innerWarpNumb ] = (offsetIter==0) 
    
    end#setIsToBeValidated
    








end#module