


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
using CUDA, Logging,Main.CUDAGpuUtils, Logging,StaticArrays, Main.IterationUtils, Main.ReductionUtils, Main.CUDAAtomicUtils,Main.MetaDataUtils




"""
this will enable iteration of metadata block
linear index is the same for each threadIdxX in a block hence the bigger in y direction is thread block the more threads will work on single metadata block
metaDataIterLoops - how many times we need to iterate over metadata with each block of threads
threadsPerBlock - number of threads in thread block
threadsPerGrid - number of threads in all thread blocks combined
maxLinIndex - maximum linear index in metadata
"""

macro metaDataWarpIter(metaDataDims,loopXMeta,loopYZMeta,ex)

    mainExp = generalizedItermultiDim(;xname=:(xMeta)
    ,yname= :(yzSpot)
    ,arrDims=metaDataDims
    ,loopXdim=loopXMeta 
    ,loopYdim=loopYZMeta
    ,isFullBoundaryCheckY=false
    ,isFullBoundaryCheckX =false
    ,yOffset = :(ydim*gridDim().x)
    ,yAdd=  :(blockIdxX()-1) 
    ,additionalActionBeforeY= :( yMeta= rem(yzSpot,$metaDataDims[2]) ; zMeta= fld(yzSpot,$metaDataDims[2]) )
    ,yCheck = :(yMeta < $metaDataDims[2] && zMeta<$metaDataDims[3] )
    ,xCheck = :(xMeta <= $metaDataDims[1])
    # ,xAdd= :(threadIdxX()-1)# to keep all 0 based
    ,is3d = false
    , ex = ex)  
    return esc(:( $mainExp))


    # return esc(quote
    # @unroll for innerIter in 0:$metaDataIterLoops-1
    #     #each thread will be responsible for single metadata block
    #     linIndex= (gridDim().x*blockDimX()*innerIter)+ (blockDimX()*(blockIdxX()-1)) + (threadIdxX())
    #     zMeta=cld(linIndex, metaXDim*metaYdim)
    #     yMeta=rem(linIndex, metaXDim*metaYdim ) 
    #     xMeta= linIndex

    #     if(linIndex<=maxLinIndex)
    #         $ex
    #     end
    # end
    # end )

end

"""
now we upload all data related to amount of data that is of our intrest 
as we need to perform basically the same work across all warps - instead on specializing threads in warp we will execute the same fynction across all warps
so warp will execute the same function just with varying data as it should be 

"""
macro loadCounters()
    return esc(quote
        @unroll for i in 1:14
           @exOnWarp i  shmemSum[threadIdxX(),i]= metaData[xMeta,yMeta+1,zMeta+1,getBeginingOfFpFNcounts()+ threadIdxX()]
        end#for
        #now in order to get offsets we need to atomially access the resOffsetCounter - we add to them total fp or fn cout so next blocks will not overwrite the 
        #area that is scheduled for this particular metadata block
        # we need to supply linear coordinate for atomicallyAddToSpot
        @exOnWarp 15 shmemSum[threadIdxX(),15]=  atomicAdd(globalFpResOffsetCounter,  ceil(metaData[xMeta,yMeta+1,zMeta+1,getBeginingOfFpFNcounts()+ 16]*1.5)  )
        @exOnWarp 16 shmemSum[threadIdxX(),16]= atomicAdd(globalFnResOffsetCounter,  ceil(metaData[xMeta,yMeta+1,zMeta+1,getBeginingOfFpFNcounts()+ 17] *1.5 )  )
         end)#quote
end #loadCounters

"""
helps in calculating linear index in meta data 
"""
function getLinearIndexInMeta(xMeta,yMeta,zMeta,metaDataDims,toAddInForthDim )
return 

end


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
         @metaDataWarpIter(metaDataDims,loopXMeta,loopYZMeta ,begin
             #now we upload all data related to amount of data that is of our intrest 
             #as we need to perform basically the same work across all warps - instead on specializing threads in warp we will execute the same fynction across all warps
             # so warp will execute the same function just with varying data as it should be 

            @loadCounters()

             sync_threads()
             ######  we need to establish is block is active at the first pass block is active simply  when total count of fp and fn is greater than 0 
            #we are adding 1 to meta y z becouse those are 0 based ...           
            @exOnWarp 1 if((shmemSum[threadIdxX(),15]) >0 )  appendToWorkQueue(workQueaue,workQueaueCounter, metaX,metaY+1,metaZ+1, false )      end 
            @exOnWarp 2 if((shmemSum[threadIdxX(),16]) >0 )  appendToWorkQueue(workQueaue,workQueaueCounter, metaX,metaY+1,metaZ+1, true )      end 
             
            @exOnWarp 3 if((shmemSum[threadIdxX(),15]) >0 ) setBlockasCurrentlyActiveInSegm(metaData, xMeta,yMeta+1,zMeta+1)    end 
            @exOnWarp 4 if((shmemSum[threadIdxX(),16]) >0 ) setBlockasCurrentlyActiveInGold(metaData, xMeta,yMeta+1,zMeta+1)     end 
 
 
            #####3set offsets
             #now we will calculate and set the result queue offsets for each offset we need to synchronize warps in order to have unique offsets 
             #we can not parallalize it more as we need to sequentially set offsets             
             
            @exOnWarp 5 @unroll for i in 0:6
                     #set fp
                   value=shmemSum[threadIdxX(),15]+ round(shmemSum[threadIdxX(),i*2+1]*1.4)
                   shmemSum[threadIdxX(),15]= value
                   metaData[xMeta,yMeta,zMeta,(getResOffsetsBeg()-1) +i*2+1  ]=value
                     end#for
 
            @exOnWarp 6 @unroll for i in 0:6
                 #set fn
                 value=shmemSum[threadIdxX(),16]+ round(shmemSum[threadIdxX(),i*2+2]*1.4) #multiply as we can have some entries potentially repeating
                 shmemSum[threadIdxX(),16]= value
                 metaData[xMeta,yMeta,zMeta,(getResOffsetsBeg()-1) +i*2+2  ]=value
                end#for
 
 
         end)# outer loop expession  )
         # probably we do not need to clear as we assign not adding values ...
         #clearSharedMemWarpLong(shmemSum, UInt8(14), Float32(0.0))
        end )
 end      


"""
will be invoked in order to iterate over the metadata  after some dilatations were already done - we need to 
    establish is block to be activated or inactivated or left as is
    if block is active it needs to be added to work queue 
    using some spare threads we will also housekeeping like for example switching active work queue etc
    we will check rescounters of border res ques and compare with old ones - if any will be grater than old we will scan for any repeating results 
        - it could be the case that neighbouring blocks concurently added the same results - in this case we need to set one of those to 0 and reduce the counter    
    we will do all by using single warp per metadata block     
"""
macro setMEtaDataOtherPasses()
    return esc(quote
    $locArr=0
    offsetIter=0
    isMaskFull=false
    @metaDataWarpIter(metaDataIterLoops, threadsPerBlock,threadsPerGrid, maxLinIndex  ,begin
    isMaskOkForProcessing=false
        #first two threads tell about wheather 
        @exOnWarp 30 resShmem[threadIdxX()+1,2,2] = isBlockFulliInGold(metaData, linIndex)
        @exOnWarp 31 resShmem[threadIdxX()+1,3,2] = isBlockToBeActivatediInGold(metaData, linIndex)
        @exOnWarp 32 resShmem[threadIdxX()+1,4,2] = isBlockCurrentlyActiveiInGold(metaData, linIndex)
        @exOnWarp 33 resShmem[threadIdxX()+1,5,2] = isBlockFullInSegm(metaData, linIndex)
        @exOnWarp 34 resShmem[threadIdxX()+1,6,2] = isBlockToBeActivatedInSegm(metaData, linIndex)
        @exOnWarp 35 resShmem[threadIdxX()+1,7,2] = isBlockCurrentlyActiveInSegm(metaData, linIndex)
        #now we will load the diffrence between old and current counter
        @unroll for i in 1:12#12 not 14 as we are intrested onli in border result queues
            @exOnWarp i begin
                #store result in registers
                #store result in registers (we are reusing some variables)
                #old count
                $locArr = getOldCount(numb, mataData,linIndex)
                #diffrence new - old 
                offsetIter= geNewCount(numb, mataData,linIndex)- $locArr
                # enable access to information is it bigger than 0 to all threads in block
                resShmem[threadIdxX()+1,i+1,3] = offsetIter>0
                end #@exOnWarp
        end#for

        #here we load data about wheather there is anything to be validated here - we save data so it can be read from the perspective of this block
        #and the blocks aroud that will want to analyze paddings
        @setIsToBeValidated()

        #now in some threads we have booleans needed for telling is mask active and in futher sixteen diffrences of counters that will tell us is there a res list that 
        #increased its  amount of value in last dilatation step if so and  this increase is in some border result list we need to  establish weather we do not have any repeating  results
        sync_threads()

        #we set information that block should be activated in gold segm
        @exOnWarp 1 if(!resShmem[threadIdxX()+1,2,2]  && (resShmem[threadIdxX()+1,3,2]  ||  resShmem[threadIdxX()+1,4,2])  )   setBlockasCurrentlyActiveInGold(metaData, linIndex)     end
        @exOnWarp 2 if(!resShmem[threadIdxX()+1,5,2]  && (resShmem[threadIdxX()+1,6,2]  ||  resShmem[threadIdxX()+1,7,2])  )   setBlockasCurrentlyActiveInSegm(metaData, linIndex)     end
        #after previous sync threads we already have the number of how much we increased number of results  relative to previous dilatation step
        #now we need to go through  those numbers and in case some of the border queues were incremented we need to analyze those added entries to establish is there 
        # any duplicate in case there will be we need to decrement counter and set the corresponding duplicated entry to 0 
        @scanForDuplicates($locArr, offsetIter) 
    end    )
        sync_threads()
        clearMainShmem(resShmem)

        clearSharedMemWarpLong(shmemSum, UInt8(14), Float32(0.0))
        $locArr=0
        offsetIter=0
    end )
end


macro setIsToBeValidated()
    return esc(quote
@unroll for i in 15:29
    @exOnWarp i setIstoBeAnalyzed(numb, mataData,linIndex) 
end#for
end )
end#setIsToBeValidated

"""
after previous sync threads we already have the number of how much we increased number of results  relative to previous dilatation step
now we need to go through  those numbers and in case some of the border queues were incremented we need to analyze those added entries to establish is there 
any duplicate in case there will be we need to decrement counter and set the corresponding duplicated entry to 0 
"""
macro scanForDuplicates(oldCount, countDiff) 
    return esc(quote
    @unroll for resQueueNumb in 1:12 #we have diffrent result queues
        @exOnWarp resQueueNumb begin
            @unroll for threadNumber in 1:32 # we need to analyze all thread id x 
                if( resShmem[warpNumb+1,resQueueNumb+1,3]) # futher actions necessary only if counter diffrence is bigger than 0 
                    if( threadIdxX()==threadNumber ) #now we need some  values that are in the registers  of the associated thread 
                        #those will be needed to know what we need to iterate over 
                        shmemSum[33,resQueueNumb]=  $oldCount 
                        shmemSum[34,resQueueNumb]=  $countDiff
                        shmemSum[35,resQueueNumb]=  linIndex
                    end# if ( threadIdxX()==warpNumb )
                    
                    sync_warp()# now we should have the required number for scanning of new values for duplicates
                    @scanForDuplicatesMainPart()
                end# resShmem[warpNumb+1,i+1,3]
            end # for warp number  
        end #@exOnWarp
    end#for 
end )
end

"""
main part of scanninf we are already on a correct warp; we already have old counter value and new counter value available in shared memory
now we need to access the result queue starting from old counter 
"""
macro scanForDuplicatesMainPart()
    return esc(quote
    #as we can analyze 32 numbers at once if the amount of new results is bigger we need to do it in multiple passes 
    @unroll for scanIter in 0: cld(shmemSum[34,i],32 )
        # here we are loading data about linear indicies of result in main bool array depending on a queue we are analyzing it will tell about gold or other pas
        if(((scanIter*32) + threadIdxX())< shmemSum[34,resQueueNumb]  )
            shmemSum[threadIdxX(),resQueueNumb] = resArray[shmemSum[33,resQueueNumb]+ (scanIter*32) + threadIdxX(),1]  
        end
        sync_warp() # now we have 32 linear indicies loaded into the shared memory
        #so we need to load some value into single value into thread and than go over all value in shared memory  
        @scanWhenDataInShmem()
    end# for scanIter 
end )
end


"""
in this spot we already have 32 (not more at least ) values in shared memory
we need to now 
"""
macro scanWhenDataInShmem()
    return esc(quote
    @unroll for tempCount in shmemSum[33,resQueueNumb] :shmemSum[33,resQueueNumb]+shmemSum[34,resQueueNumb]
        #now we need to make sure that we are not at spot whre this value is legitimite - so this is first occurence
        if(tempCount!=shmemSum[33,resQueueNumb]+ (scanIter*32) + threadIdxX()  )
            #finally we iterate over all values in any given thread and compare to associated value in shared memory
            if(resArray[tempCount,1]  == shmemSum[threadIdxX(),resQueueNumb] )
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
    resArray[tempCount,1]=0
    decrCounterByOne(numb, mataData,shmemSum[35,resQueueNumb])
    
end )
end




# isBlockFull(metaData, linIndex)
# isBlockToBeActivated(metaData, linIndex)


# HFUtils.clearMainShmem(resShmem)
#         # first we check weather next block is viable for processing
#         @unroll for zIter in 1:6
 
#           ----------- what is crucial those actions will be happening on diffrent threads hence when we will reduce it we will know results from all        
     
#             #we will iterate over all padding planes below way to calculate the next block in all dimensions not counting oblique directions
#             @ifXY 1 zIter isMaskOkForProcessing = ((currBlockX+UInt8(zIter==1)-UInt8(zIter==2))>0)
#             @ifXY 2 zIter @inbounds isMaskOkForProcessing = (currBlockX+UInt8(zIter==1)-UInt8(zIter==2))<=metadataDims[1]
#             @ifXY 3 zIter @inbounds isMaskOkForProcessing = (currBlockY+UInt8(zIter==3)-UInt8(zIter==4))>0
#             @ifXY 4 zIter @inbounds isMaskOkForProcessing = (currBlockY+UInt8(zIter==3)-UInt8(zIter==4))<=metadataDims[2]
#             @ifXY 5 zIter @inbounds isMaskOkForProcessing = (currBlockZ+UInt8(zIter==5)-UInt8(zIter==6))>0
#             @ifXY 6 zIter @inbounds isMaskOkForProcessing = (currBlockZ+UInt8(zIter==5)-UInt8(zIter==6))<=metadataDims[3]
#             @ifXY 7 zIter @inbounds isMaskOkForProcessing = !metaData[currBlockX+UInt8(zIter==1)-UInt8(zIter==2)
#                                                             ,(currBlockY+UInt8(zIter==3)-UInt8(zIter==4))
#                                                             ,(currBlockZ+UInt8(zIter==5)-UInt8(zIter==6)),isPassGold+3]#then we need to check weather mask is already full - in this case we can not activate it 
#             #now we check are all true 
#                  ----------- this can be done by one of the reduction macros    

#            offset = UInt8(1)
#             @ifY zIter begin 
#                 while(offset <UInt8(8)) 
#                     @inbounds isMaskOkForProcessing =  isMaskOkForProcessing & shfl_down_sync(FULL_MASK, isMaskOkForProcessing, offset)
#                     offset<<= 1
#                 end #while
#             end# @ifY 
#         #here is the information wheather we want to process next block
#         @ifXY 1 zIter @inbounds resShmem[2,zIter+1,2] = isMaskOkForProcessing
#          end#for zIter   
                
#          sync_threads()#now we should know wheather we are intrested in blocks around
       
   
            
            
#         # ################################################################################################################################ 
#         #checking is there anything in the padding plane - so we basically do (most of reductions)
#         #values stroing in local registers is there anything in padding associated # becouse we will store it in one int we can pass it at one move of registers
#         locArr=0 #reset for reuse
#                ----------- this was created for cubic 32x32x32 block where one plane of threads can analyze all paddings 
#                    ----------- as in variable size thread blocks some of threads when processing padding will have nothing to do we can think so it will work in this time on the  isMaskForProcessing from above
#         locArr|= resShmem[ 34 ,threadIdxX() , threadIdxY() ] << 1 #RIGHT
#         locArr|= resShmem[1 ,threadIdxX() , threadIdxY()] << 2 #LEFT
#         locArr|= resShmem[threadIdxX() ,34 ,threadIdxY() ] << 3 #ANTERIOR
#         locArr|=  resShmem[ threadIdxX(),1 , threadIdxY()] << 4 #POSTERIOR
#         locArr|= resShmem[ threadIdxX() , threadIdxY() ,1] << 5 #TOP
#         locArr|= resShmem[ threadIdxX() , threadIdxY() ,34] << 6 #BOTTOM

#    ----------- this reduction can be done probably together with reduction from step above        
#                 #we need to reduce now  the values  of padding vals to establish weather there is any true there if yes we put the neighbour block to be active 
#                     #reduction                   
#                     offset = UInt8(1)
#                     while(offset <32) 
#                         #we load first value from nearby thread 
#                         shuffled = shfl_down_sync(FULL_MASK, locArr, offset)
#                         #we loop over  bits and updating we are intrested weather there is any positive so we use or
#                         @unroll for zIter::UInt8 in UInt8(1):UInt8(6)
#                             locArr|= @inbounds ((shuffled>>zIter & 1) | @inbounds  (locArr>>zIter & 1) ) <<zIter
#                         end#for    
#                         #isMaskOkForProcessing = (isMaskOkForProcessing | 
#                         offset<<= 1
#                     end

#                     @unroll for zIter::UInt8 in UInt8(1):UInt8(6)
#                         @ifX 1  resShmem[zIter+1,threadIdxY()+1,3]=  @inbounds  (locArr>>zIter & 1)
#                         #@ifX 1 CUDA.@cuprint " resShmem[zIter+1,threadIdxX()+1,3]   $(resShmem[zIter+1,threadIdxX()+1,3] )   locArr $(locArr) \n" 
#                     end#for  
                             
#              sync_threads()#now we have partially reduced values marking wheather we have any true in padding         
#                   #  # we get full reductions
#             @unroll for zIter::UInt8 in UInt8(1):UInt8(6)
#                 if(resShmem[2,zIter+1,2] )
#                 offset = UInt8(1)
#                 if(UInt8(threadIdxY())==zIter)
#                     while(offset <32)                        
#                         @inbounds  resShmem[zIter+1,threadIdxX()+1,3] = (resShmem[zIter+1,threadIdxX()+1,3] | shfl_down_sync(FULL_MASK,resShmem[zIter+1,threadIdxX()+1,3], offset))
#                         offset<<= 1
#                     end#while
#                 end#if    
#                 end#if                          
#             end#for

#             sync_threads()#now we have fully reduced in resShmem[zIter+1,1+1,3]= resShmem[zIter+1,2,3]
    
                
                
                
                
#                     #updating metadata
#     if(resShmem[2,primaryZiter+1,2] && resShmem[primaryZiter+1,2,3] )   
#         @ifXY 2 primaryZiter @inbounds  metaData[(currBlockX+(primaryZiter==1)-(primaryZiter==2)),(currBlockY+(primaryZiter==3)-(primaryZiter==4)),(currBlockZ+(primaryZiter==5)-(primaryZiter==6)),isPassGold+1]= true
#     end#if
#     sync_warp()


    
    
  end #MetadataAnalyzePass
