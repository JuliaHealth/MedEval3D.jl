


"""
For first metadata pass we need already cut the area o boolean arrays that we are intrested in and portion of metadata array that we are intrested in 
this way indexing will be simpler we will start from 0 and we will reduce memor usage


3) first metadata pass we add to the metadata offset where each block will put its main results and padding results - so all will be stored in result quueue but in diffrent spots
    we need to make ques for paddings longer than number of possible results becouse of possible modifications from neighbouring blocks that can happen simultanously
    we will establish this offsets using atomics- at this pass we will also prepare first work queue with indicies of metadata blocks and booleans indicating is it related to gold pass dilatation step or other pass

5) we do the metadata pass we analyze only those blocks that are in the borders of intrests - max min x y z was specified in step 1 , we check weather block is set to be activated
 and in case it is not full we will make it active , if the block is set as active we just add it to work queue  that is appropriate to next iteration
  - we need also to scan  the border result ques if there is any duplicate result - if so we set it to zeros and we reduce the border result counter
        of course we check it only in case new counter is bigger than old counter 


        so 1) we check metadata when metadata comply with our predicate described above we atomically increase local shared memory work queue counter
we synchronize
        -we proceed if local workqueue counter is greater than 0 
        2)  we add the local workqueue counter to global keep old value as an offset in shared memory 
        3) sync threads,  add to the work queue the data from registers of those threads that met the predicate
"""

module  MetadataAnalyzePass     
using CUDA, Logging,Main.CUDAGpuUtils, Main.ResultListUtils,Main.WorkQueueUtils,Main.ScanForDuplicates, Logging,StaticArrays, Main.IterationUtils, Main.ReductionUtils, Main.CUDAAtomicUtils,Main.MetaDataUtils, Main.ScanForDuplicates
export @metaDataWarpIter, @loadCounters,@analyzeMetadataFirstPass, @checkIsActiveOrFullOr,@setIsToBeActive




"""
this will enable iteration of metadata block
linear index is the same for each threadIdxX in a block hence the bigger in y direction is thread block the more threads will work on single metadata block

"""
macro metaDataWarpIter(metaDataDims,loopWarpMeta,metaDataLength,ex)

    return  esc(quote
    isInRange = true
    linIdexMeta = UInt32(0)

    # gridDimX()*blockDimX()

    @unroll for j in 0:($loopWarpMeta-1)
        linIdexMeta= ((blockIdxX()-1)*$loopWarpMeta*blockDimX())+j*blockDimX()+ threadIdxX()-1
        # linIdexMeta= ((blockIdxX()-1)*(gridDimX())*blockDimX()*$loopWarpMeta)+j*blockDimX()+ threadIdxX()
        isInRange=(linIdexMeta<$metaDataLength)
        xMeta= rem(linIdexMeta,$metaDataDims[1])
        zMeta= fld((linIdexMeta),$metaDataDims[1]*$metaDataDims[2])
        yMeta= fld((linIdexMeta-((zMeta*$metaDataDims[1]*$metaDataDims[2] ) + xMeta )),$metaDataDims[1])
      $ex
    end 

    end)


    # mainExp = generalizedItermultiDim(;xname=:(xMeta)
    # ,yname= :(yzSpot)
    # ,arrDims=metaDataDims
    # ,loopXdim=loopXMeta 
    # ,loopYdim=loopYZMeta
    #  ,yOffset = :(ydim*gridDim().x)
    # ,yAdd=  :(blockIdxX()-1) 
    # ,additionalActionBeforeY= :( yMeta= rem(yzSpot,$metaDataDims[2]) ; zMeta= fld(yzSpot,$metaDataDims[2]) )
    # ,additionalActionBeforeX= :( isInRange = ( yMeta < $metaDataDims[2] && zMeta<$metaDataDims[3] && xMeta <= $metaDataDims[1]  ) )
    # ,nobundaryCheckX=true
    # , nobundaryCheckY=true
    # , nobundaryCheckZ =true
    # # ,yCheck = :(yMeta < $metaDataDims[2] && zMeta<$metaDataDims[3] )
    # # ,xCheck = :(xMeta <= $metaDataDims[1])
    # # ,xAdd= :(threadIdxX()-1)# to keep all 0 based
    # ,is3d = false
    # , ex = ex)  
    # return esc(:( $mainExp))
end


"""
now we upload all data related to amount of data that is of our intrest 
as we need to perform basically the same work across all warps - instead on specializing threads in warp we will execute the same fynction across all warps
so warp will execute the same function just with varying data as it should be 

"""
macro loadCounters()
    return esc(quote
        @unroll for i in 1:14
           @exOnWarp i begin 
            if(isInRange) 
                shmemSum[threadIdxX(),i]= @accMeta(getBeginingOfFpFNcounts()+ i)
            end   
           end
        end#for
        
        #now in order to get offsets we need to atomially access the resOffsetCounter - we add to them total fp or fn cout so next blocks will not overwrite the 
        #area that is scheduled for this particular metadata block
        # we need to supply linear coordinate for atomicallyAddToSpot
        @exOnWarp 15 begin 
            if(isInRange)
            # count = @accMeta(getBeginingOfFpFNcounts()+ 16)
            #     if(count>0)     
            #         shmemSum[threadIdxX(),15]= atomicAdd(globalFpResOffsetCounter,  ceil(count*1.5)  )+1
            #     else
            #         shmemSum[threadIdxX(),15]= 0
            #     end    
            end
        end


        # @exOnWarp 16 begin 
        #     if(isInRange) 
        #     #count = @accMeta(getBeginingOfFpFNcounts()+ 17)
        #         if(count>0)     
        #             shmemSum[threadIdxX(),16]= atomicAdd(globalFnResOffsetCounter,  ceil( count*1.5 )  )+1
        #         else
        #             shmemSum[threadIdxX(),16]= 0
        #         end    
        #     end   
        # end
            
        end)#quote
end #loadCounters


 """
 we analyze metadate as described above 
 minX, minY,minZ - minimal indexes of metadata that holds all of the data that is of intrest to us
 maxX,maxY,maxZ  - maximal indexes of metadata that holds all of the data that is of intrest to us
 metaData - global memory data structure that we analyze
 shmemSum - shared memory used primary for reductions 
 globalFpResOffsetCounter, globalFnResOffsetCounter  - counters accessed atomically that points where we want to set the  results from this metadata block
 workQueaueA, workQueaueAcounter
 tobeEx - some register boolean that we will reuse
 """
 macro analyzeMetadataFirstPass()
         return esc(quote
         # we need to iterate over all metadata blocks with checks so the blocks can not be  outside the area of intrest defined by  minX, minY,minZ and maxX,maxY,maxZ
         @metaDataWarpIter(metaDataDims,loopWarpMeta,metaDataLength,begin
             #now we upload all data related to amount of data that is of our intrest 
             #as we need to perform basically the same work across all warps - instead on specializing threads in warp we will execute the same fynction across all warps
             # so warp will execute the same function just with varying data as it should be 

           @loadCounters() 

            sync_threads()

        #     #  ######  we need to establish is block is active at the first pass block is active simply  when total count of fp and fn is greater than 0 
        #     # #we are adding 1 to meta y z becouse those are 0 based ...           

        #     @ifY 1 if(shmemSum[threadIdxX(),15]>0 && isInRange) begin  
        #             appendToWorkQueue(workQueaue,workQueaueCounter, xMeta,yMeta+1,zMeta+1, 0 ) 
        #         end   
        #     end     
        #     @ifY 2 if(shmemSum[threadIdxX(),16]>0 && isInRange) begin 
        #          appendToWorkQueue(workQueaue,workQueaueCounter, xMeta,yMeta+1,zMeta+1, 1 ) end  
        #         end      
            
        #     @exOnWarp 3 if((shmemSum[threadIdxX(),15]) >0 && isInRange) setBlockasCurrentlyActiveInSegm(metaData, xMeta+1,yMeta+1,zMeta+1)    end 
        #     @exOnWarp 4 if((shmemSum[threadIdxX(),16]) >0 && isInRange) setBlockasCurrentlyActiveInGold(metaData, xMeta+1,yMeta+1,zMeta+1)     end 
 
 
        #     #####3set offsets
        #      #now we will calculate and set the result queue offsets for each offset we need to synchronize warps in order to have unique offsets 
        #      #we can not parallalize it more as we need to sequentially set offsets             
             
        #     @exOnWarp 5 begin if((shmemSum[threadIdxX(),15]) >0 && isInRange) @unroll for i in 0:6
        #              #set fp
        #            value=floor(shmemSum[threadIdxX(),15])+1
        #            if(i>0)
        #             value+= ceil(shmemSum[threadIdxX(),((i-1)*2+1)]*1.45)
        #            end
        #            shmemSum[threadIdxX(),15]= value
        #            @setMeta(((getResOffsetsBeg()-1) +i*2+1) ,value)
        #              end#for
        #             end #if
        #         end
 
        #     @exOnWarp 6 begin if((shmemSum[threadIdxX(),16]) >0 && isInRange) @unroll for i in 0:6
        #          #set fn
        #          value=shmemSum[threadIdxX(),16]
        #          if(i>0)
        #             value+= ceil(shmemSum[threadIdxX(),((i-1)*2+2)]*1.45)+1 #multiply as we can have some entries potentially repeating
        #          end
        #          shmemSum[threadIdxX(),16]= value
        #          @setMeta(((getResOffsetsBeg()-1) +i*2+2) ,value)
        #         end#for
        #     end#if
        # end
 
 
         end)# outer loop expession  )
         # probably we do not need to clear as we assign not adding values ...
         #clearSharedMemWarpLong(shmemSum, UInt8(14), Float32(0.0))
        end )
 end      
"""
establish is the  block  is active full or be activated, and we are saving this information into surcehmem
"""
macro checkIsActiveOrFullOr()
    return esc(quote
        @exOnWarp 30 if(isInRange) sourceShmem[(threadIdxX())] = @accMeta(getFullInGoldNumb() ) end#  isBlockFulliInGold(metaData, xMeta,yMeta+1,zMeta+1)
        @exOnWarp 31 if(isInRange)   sourceShmem[(threadIdxX())+33] = @accMeta(getIsToBeActivatedInGoldNumb() ) end # isBlockToBeActivatediInGold(metaData, xMeta,yMeta+1,zMeta+1)
        @exOnWarp 32 if(isInRange)  sourceShmem[(threadIdxX())+33*2] = @accMeta(getActiveGoldNumb() ) end # isBlockCurrentlyActiveiInGold(metaData, xMeta,yMeta+1,zMeta+1)
       
        @exOnWarp 33 if(isInRange) sourceShmem[(threadIdxX())+33*3] = @accMeta(getFullInSegmNumb()) end # isBlockFullInSegm(metaData, xMeta,yMeta+1,zMeta+1)
        @exOnWarp 34 if(isInRange) sourceShmem[(threadIdxX())+33*4] = @accMeta(getIsToBeActivatedInSegmNumb() ) end # isBlockToBeActivatedInSegm(metaData, xMeta,yMeta+1,zMeta+1)
        @exOnWarp 35 if(isInRange) sourceShmem[(threadIdxX())+33*5] = @accMeta(getActiveSegmNumb()) end # isBlockCurrentlyActiveInSegm(metaData, xMeta,yMeta+1,zMeta+1)
end)#quote
end#checkIsActiveOrFullOr

"""
given data in sourceShmem loaded by checkIsActiveOrFullOr() we will  mark the block as active  ( or not) 
    and if is to be active add it to work queue
"""
macro setIsToBeActive()
    return esc(quote
        @exOnWarp 1 if(!sourceShmem[(threadIdxX())]  && (sourceShmem[(threadIdxX())+33]  ||  sourceShmem[(threadIdxX())+33*2]) &&isInRange  )  
                        @setMeta(getActiveGoldNumb(),1)
                        appendToWorkQueue(workQueaue,workQueaueCounter, xMeta,yMeta+1,zMeta+1, 1 )
                    end
        @exOnWarp 2 if(!sourceShmem[(threadIdxX())+33*3]  && (sourceShmem[(threadIdxX())+33*4]  ||  sourceShmem[(threadIdxX())+33*5]) &&isInRange ) 
                        @setMeta(getActiveSegmNumb(),1)
                        appendToWorkQueue(workQueaue,workQueaueCounter, xMeta,yMeta+1,zMeta+1, 0 )             
            end
    end)#quote

end    






    """
    will be invoked in order to iterate over the metadata  after some dilatations were already done - we need to 
        establish is block to be activated or inactivated or left as is
        if block is active it needs to be added to work queue 
        using some spare threads we will also housekeeping like for example switching active work queue etc
        we will check rescounters of border res ques and compare with old ones - if any will be grater than old we will scan for any repeating results 
            - it could be the case that neighbouring blocks concurently added the same results - in this case we need to set one of those to 0 and reduce the counter    
        we will do all by using single warp per metadata block     
        globalCurrentFpCount, globalCurrentFnCount - representing current number of already covere fp and fns
    """
    macro setMEtaDataOtherPasses(locArr,offsetIter,iterThrougWarNumb)
        return esc(quote
        $locArr=0
        $offsetIter=0
        isMaskFull=false
        @metaDataWarpIter(metaDataDims,loopWarpMeta,metaDataLength, begin
        isMaskOkForProcessing=false
            #first we will check is block full active or be activated and we will set later on this basis what blocks should be put to work queue
             @checkIsActiveOrFullOr() 
          
            #now we need to go through  those numbers and in case some of the border queues were incremented we need to analyze those added entries to establish is there 
            # any duplicate in case there will be we need to decrement counter and set the corresponding duplicated entry to 0 
            #here we load data about wheather there is anything to be validated here - we save data so it can be read from the perspective of this block
            # #and the blocks aroud that will want to analyze paddings
            @loadAndScanForDuplicates(iterThrougWarNumb,$locArr,$offsetIter,localOffset)



            #we set information that block should be activated in gold  and segm
             @setIsToBeActive() 

        end    )
        sync_threads()

        #now we add to the global variables all of the fps and fns after corrections for duplicates
        @ifXY 1 1 begin 
            # if(xMeta==1 && yMeta==0 && zMeta==0)
            #     CUDA.@cuprint """  valuee fp $(alreadyCoveredInQueues[1]+ alreadyCoveredInQueues[3]+ alreadyCoveredInQueues[5]+ alreadyCoveredInQueues[7]+ alreadyCoveredInQueues[9]+ alreadyCoveredInQueues[11]+ alreadyCoveredInQueues[13]) 
            #     alreadyCoveredInQueues[1] $(alreadyCoveredInQueues[1]) alreadyCoveredInQueues[3] $(alreadyCoveredInQueues[3]) alreadyCoveredInQueues[5] $(alreadyCoveredInQueues[5]) alreadyCoveredInQueues[7] $(alreadyCoveredInQueues[7]) alreadyCoveredInQueues[9] $(alreadyCoveredInQueues[9]) alreadyCoveredInQueues[11] $(alreadyCoveredInQueues[11]) alreadyCoveredInQueues[13] $(alreadyCoveredInQueues[13]) 
                
            #     \n"""
            # end  
            atomicAdd(globalCurrentFpCount, alreadyCoveredInQueues[1]+ alreadyCoveredInQueues[3]+ alreadyCoveredInQueues[5]+ alreadyCoveredInQueues[7]+ alreadyCoveredInQueues[9]+ alreadyCoveredInQueues[11]+ alreadyCoveredInQueues[13]) 
        end
            @ifXY 2 1 atomicAdd(globalCurrentFnCount, alreadyCoveredInQueues[2]+ alreadyCoveredInQueues[4]+ alreadyCoveredInQueues[6]+ alreadyCoveredInQueues[8]+ alreadyCoveredInQueues[10]+ alreadyCoveredInQueues[12]+ alreadyCoveredInQueues[14]) 


            sync_threads()
    


            @metaDataWarpIter(metaDataDims,loopWarpMeta,metaDataLength, begin
            #now we need to set old caounters to the value of new counters so at next dilatation we will count only new values ...
            for i in 1:14
                @exOnWarp (i+37) @setMeta((getOldCountersBeg() +i),@accMeta(getNewCountersBeg() +i))
            end  
            end)   


            #clear used shmem - we used linear indicies so we can clear only those used
            for i in 0:30
                @exOnWarp i resShmem[(threadIdxX())+(i)*33]= false
             end
             for i in 0:8#was 6
                @exOnWarp (i+15) sourceShmem[(threadIdxX())+(i)*33]= false
             end   
             for i in 1:14
                @exOnWarp (i+23) shmemSum[threadIdxX(),i]= 0
             end

            $locArr=0
            $offsetIter=0
            sync_threads()

        end )
    end






    
    
  end #MetadataAnalyzePass
