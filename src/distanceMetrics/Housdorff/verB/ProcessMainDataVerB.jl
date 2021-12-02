module ProcessMainDataVerB
using CUDA, Logging,Main.CUDAGpuUtils,Main.WorkQueueUtils, Logging,StaticArrays,Main.MetaDataUtils, Main.IterationUtils, Main.ReductionUtils, Main.CUDAAtomicUtils,Main.MetaDataUtils, Main.ResultListUtils
export @dilatateHelper,getDir,@validateData, @executeDataIterWithPadding, @loadMainValues,setNextBlockAsIsToBeActivated,@paddingProcessCombined,calculateLoopsIter,@processMaskData, @paddingIter,@processPadding


                
"""
 validates data is of our intrest               
"""                
macro validateData(isGold,xMeta,yMeta,zMeta,iterNumb,mainArr,refArr,targetArr)
    return esc(quote
   #first we will load data from target arr so we can be sure that we are not ovewriting sth already written by diffrent thread block
   locArr = @inbounds($targetArr[($xMeta-1)*dataBdim[1]+ threadIdxX(),($yMeta-1)*dataBdim[2]+ threadIdxY(),$zMeta])
   # here we are anaylyzing only main part of the  data block paddings will be analyzed separately
   @unroll for bitIter in 1:32
       resShemVal = isBit1AtPos(@inbounds(resShmemblockData[threadIdxX(),threadIdxY()]), bitIter)
       inTarget = isBit1AtPos(locArr, bitIter)
       #later usefull to establish is mask full
       isMaskFull= (resShemVal && isMaskFull)
    #    #so we have voxel that was not yet covered earlier, is covered now and is in refrence arr - so one where we establish is sth covered 
       if(resShemVal && !inTarget)
           # we need reference array value to check weather we had covered anything new from other arrayin this iteration
            if($refArr[($xMeta-1)*dataBdim[1]+ threadIdxX(),($yMeta-1)*dataBdim[2]+ threadIdxY(),($zMeta-1)*dataBdim[3]+ bitIter  ] == numberToLooFor)
                @addResult(metaData
                ,$xMeta,$yMeta ,$zMeta,resList
                ,(($xMeta-1)*dataBdim[1]+ threadIdxX())
                ,(($yMeta-1)*dataBdim[2]+ threadIdxY())
                ,($zMeta-1)*dataBdim[3]+ bitIter
                ,getDir(shmemblockData,bitIter,dataBdim)
                ,$iterNumb
                ,getIndexOfQueue(threadIdxX(),threadIdxY(),bitIter,dataBdim,(1-$isGold))
                ,metaDataDims
                ,mainArrDims
                ,$isGold)
            end
       end  
   end 
   
    end)


    
 end  #validateData                  


"""
   loads main values from analyzed array into shared memory and to locArr - which live in registers   
   it all works under the assumption that x and y dimension of the thread block and data block is the same           
"""                
                
macro loadMainValues(mainArr,xMeta,yMeta,zMeta)
    return esc(quote
    # we already loaded locArr and shmemblockData in executeIterPadding

    #now immidiately we can go with dilatation up and down and save it to res shmem we are not modyfing  locArr
    @inbounds resShmemblockData[threadIdxX(),threadIdxY()]=@bitDilatate(locArr)
    # now if we have values in first or last bit we need to modify appropriate spots in the shmemPaddings
    @inbounds shmemPaddings[threadIdxX(),threadIdxY(),1]=isBit1AtPos(locArr,1)#top
    @inbounds shmemPaddings[threadIdxX(),threadIdxY(),2]=isBit1AtPos(locArr,dataBdim[3])#bottom
    #now we will  do left - right dilatations howvewer we must be sure that we checked boundary conditions 
    
    #left
    @dilatateHelper((threadIdxX()==1), 3,bitPos,threadIdxY(),(-1), (0))

    #right
    @dilatateHelper((threadIdxX()==dataBdim[1]), 4,bitPos,threadIdxY(),(1), (0))

    #  #posterior
    @dilatateHelper((threadIdxY()==1), 5,threadIdxX(), bitPos,(0), (-1))

    #   #anterior 
    @dilatateHelper((threadIdxY()==dataBdim[2]), 6,threadIdxX(), bitPos,(0), (1))
    sync_threads()
    #now we need to persist the paddings still becouse its size is up to 32 by 32 we need to iterate over y dimension
    for iterY in 1:inBlockLoopXZIterWithPadding
        if((threadIdxY()+iterY)<=dataBdim[1]  )
            #we are reusing offsetIter
            offsetIter=0
            for bitPos in 1:6
                @setBitTo(offsetIter,bitPos, shmemPaddings[threadIdxX(),(threadIdxY()+iterY),bitPos ])
            end
            @inbounds paddingStore[$xMeta,$yMeta,$zMeta,threadIdxX(),(threadIdxY()+iterY)]= offsetIter
        end
    end
end) #quote              
end #loadMainValues


"""
helper macro to iterate over the threads and given their position - checking edge cases do appropriate dilatations ...
    predicate - indicates what we consider border case here 
    paddingPos= integer marking which padding we are currently talking about (top? bottom? anterior ? ...)
    padingVariedA, padingVariedB - eithr bitPos threadid X or Y depending what will be changing in this case
    
    normalXChange, normalYchange - indicating which wntries we are intrested in if we are not at the boundary so how much to add to x and y thread position

"""
macro dilatateHelper(predicate, paddingPos, padingVariedA, padingVariedB,normalXChange, normalYchange)
    return esc(quote
        if($predicate)
            for bitPos in 1:32
                shmemPaddings[$padingVariedA,$padingVariedB,$paddingPos]=isBit1AtPos(locArr,bitPos)
                #we need to mark weather there is anything in padding so we will mark block as to be activates 
                if(shmemPaddings[$padingVariedA,$padingVariedB,$paddingPos]) 
                    isAnythingInPadding[$paddingPos]= true 
                end
            end
        else
          resShmemblockData[threadIdxX(),threadIdxY()]= @bitPassOnes(resShmemblockData[threadIdxX(),threadIdxY()],shmemblockData[threadIdxX()+($normalXChange),threadIdxY()+$(normalYchange)]   )
        end 
    end)#quote
end




"""
now in case we  want later to establish source of the data - would like to find the true distances  not taking the assumption of isometric voxels
we need to store now data from what direction given voxel was activated what will later gratly simplify the task of finding the true distance 
we will record first found true voxel from each of six directions 
                 top 6 
                bottom 5  
                left 2
                right 1 
                anterior 3
                posterior 4
"""
function getDir(shmemblockData,bitIter,dataBdim)::UInt8
    return if((bitIter-1)>0 && isBit1AtPos(@inbounds(shmemblockData[threadIdxX(),threadIdxY()]), bitIter-1) ) 
                6
            elseif(((bitIter)<dataBdim[3]) && isBit1AtPos(@inbounds(shmemblockData[threadIdxX(),threadIdxY()]), bitIter+1) ) 
                5
            elseif((threadIdxX()-1>0) && isBit1AtPos(@inbounds(shmemblockData[threadIdxX()-1,threadIdxY()]), bitIter) )
                2
            elseif((threadIdxX()<dataBdim[1]) && isBit1AtPos(@inbounds(shmemblockData[threadIdxX()+1,threadIdxY()]), bitIter) )
                1
            elseif((threadIdxY()-1>0) && isBit1AtPos(@inbounds(shmemblockData[threadIdxX(),threadIdxY()-1]), bitIter) )
                3
            else 
                4            
            end
end#getDir

"""
collects all needed functions to analyze given data blocks 
- so it loads data from main array (what is main and reference array depends on is it a gold pass or other pass)
then 
"""
macro executeDataIter(mainArrDims
    ,mainArr,refArr,xMeta,yMeta,zMeta,isGold,iterNumb)
    return esc(quote
  ############## upload data
  @ifY 1 if(threadIdxX()<15) areToBeValidated[threadIdxX()] =metaData[($xMeta+1),($yMeta+1),($zMeta+1),(getIsToBeAnalyzedNumb() +threadIdxX())]  end 
  @loadMainValues($mainArr,$xMeta,$yMeta,$zMeta)                                       
  sync_threads()
  ########## check data aprat from padding
  #can be skipped if we have the block with already all results analyzed 
  if(areToBeValidated[14-$isGold])
      @validateData($mainArrDims, $inBlockLoopX,$inBlockLoopY,$inBlockLoopZ,$mainArr,$refArr,$xMeta,$yMeta,$zMeta,$isGold,$iterNumb) 
  end                  
  
  #now we need to establish are we full here; and whether neighbours are to be activated
  isMaskFull =syncThreadsAnd(isMaskFull)
  @ifXY 1 1 setNextBlockAsIsToBeActivated(isAnythingInPadding[1] ,-1,0,0,$xMeta,$yMeta,$zMeta,$isGold,  metaData, metaDataDims)
  @ifXY 2 1 setNextBlockAsIsToBeActivated(isAnythingInPadding[2] ,1,0,0,$xMeta,$yMeta,$zMeta,$isGold,  metaData, metaDataDims)
  @ifXY 3 1 setNextBlockAsIsToBeActivated(isAnythingInPadding[3] ,0,-1,0,$xMeta,$yMeta,$zMeta,$isGold,  metaData, metaDataDims)
  @ifXY 4 1 setNextBlockAsIsToBeActivated(isAnythingInPadding[4] ,0,1,0,$xMeta,$yMeta,$zMeta,$isGold,  metaData, metaDataDims)
  @ifXY 5 1 setNextBlockAsIsToBeActivated(isAnythingInPadding[5] ,0,0,-1,$xMeta,$yMeta,$zMeta,$isGold,  metaData, metaDataDims)
  @ifXY 6 1 setNextBlockAsIsToBeActivated(isAnythingInPadding[6] , 0,0,1,$xMeta,$yMeta,$zMeta,$isGold,  metaData, metaDataDims)
  @ifXY 7 1 if(isMaskFull) metaData[$xMeta,$yMeta,$zMeta,getFullInSegmNumb()-$isGold]=1  end

end)
end#executeDataIterWithPadding


"""
collects all needed functions to analyze given data blocks 
- so it loads data from main array (what is main and reference array depends on is it a gold pass or other pass)
then 
"""
macro executeIterPadding(mainArr,refArr,xMeta,yMeta,zMeta,isGold,iterNumb)
    return esc(quote
    # we already gone through dilatation and metadata analysis so now we are analysing paddings from previous loop
    #first we need to load data from paddings of surrounding blocks and  push it into shared memory
    #yet we need to check weather in given direction there is some block so if we are on edge ?

    #order of paddings for reference 1)top, 2)bottom, 3)left 4)right , 5)anterior, 6)posterior
    
    #for example here we look to the right  block so we are putting data in our local right padding from left padding of neighpbouring block
    loadToshmemPaddings(xMeta,yMeta,zMeta, 1,0,0, 3, 4)

    loadToshmemPaddings(xMeta,yMeta,zMeta, -1,0,0, 4, 3)
    loadToshmemPaddings(xMeta,yMeta,zMeta, 0,1,0, 6, 5)
    loadToshmemPaddings(xMeta,yMeta,zMeta, 0,-1,0, 5, 6)
    loadToshmemPaddings(xMeta,yMeta,zMeta, 0,0,1, 2, 1)
    loadToshmemPaddings(xMeta,yMeta,zMeta, 0,0,-1, 1, 2)
    #so now we have data loaded from surrounding blocks about their paddings we may now modify accordingly current block
    #by construction one thread will neeed to load just one integer into its registers and to resShmemblockData
    locArr = $mainArr[($xMeta-1)*dataBdim[1]+ threadIdxX(),($yMeta-1)*dataBdim[2]+ threadIdxY(),$zMeta]
    @inbounds shmemblockData[threadIdxX(),threadIdxY()] = locArr
    sync_threads()
    # now we have data loaded about what is already in dilatation array what was marked from paddings of neighbouring blocks
    # so we need to add dilatation and if needed result to result list 


    


    end)
end#executeDataIterWithPadding

"""
# now we have data loaded about what is already in dilatation array what was marked from paddings of neighbouring blocks
# so we need to add dilatation and if needed result to result list
predicate - checks weather we have set bit in position we are intrested in 
"""
macro dilatateAndAddResFromPadding(predicate, )
    return esc(quote

    end)
end


"""
helper function for executeIterPadding - we will check is next block in each direction is in metadata 
    xMetaChange,yMetaChange,zMetaChange - whrere to look for block of interest relative to current position
    bitOfIntrest - which bit from paddingStore we are intrested in in a block we are analyzing now 
    shmemPaddingTargetNumb - where we should write the data from surrounding block into our local shmem padding
    """
macro loadToshmemPaddings(xMeta,yMeta,zMeta, xMetaChange,yMetaChange,zMetaChange, bitOfIntrest, shmemPaddingTargetNumb)
    return esc(quote
    #we need to be sure that such block exists
    if( ($xMeta)+$xMetaChange<=metaDataDims[1]  && ($yMeta)+$yMetaChange>0 && ($yMeta)+$yMetaChange<=metaDataDims[2] && ($zMeta)+$zMetaChange>0 && ($zMeta)+$zMetaChange<=metaDataDims[3])
        @unroll for iterY in 1:inBlockLoopXZIterWithPadding
            if((threadIdxY()+iterY)<=dataBdim[1]  )
                #so we are intrested only in given bit from neighbouring block
                booll = isBit1AtPos(paddingStore[$xMeta+$xChange,$yMeta+$yChange,$zMeta+$zChange],bitOfIntrest)
                shmemPaddings[threadIdxX(),(threadIdxY()+iterY),shmemPaddingTargetNumb ]=booll
            end    
        end#for    
    end#if such block exists    
    offsetIter
    end)
end






"""
combines above function to make it more convinient to call
"""
macro paddingProcessCombined(loopX,loopY,maxXdim, maxYdim,a,b,c , xMetaChange,yMetaChange,zMetaChange, mainArr,refArr, dir,iterNumb,queueNumber,xMeta,yMeta,zMeta,isGold)
 
    for iterY in 1:inBlockLoopXZIterWithPadding
 
    #   function paddingProcessCombined(loopX,loopY,maxXdim, maxYdim,a,b,c , xMetaChange,yMetaChange,zMetaChange, mainArr,refArr, dir,iterNumb,queueNumber,xMeta,yMeta,zMeta,isGold,resShmem,dataBdim,metaData,resList,resListIndicies,maxResListIndex,metaDataDims )
    return esc(quote
    @paddingIter($loopX,$loopY,$maxXdim, $maxYdim, begin
        if(resShmem[$a,$b,$c])
                #CUDA.@cuprint "meta bounds checks $($xMeta+$xMetaChange>0) $($xMeta+$xMetaChange<=metaDataDims[1]) $($yMeta+$yMetaChange>0) $($yMeta+$yMetaChange<=metaDataDims[2])  $($zMeta+$zMetaChange>0) $($zMeta+$zMetaChange<=metaDataDims[3]) \n "
                @inbounds isAnythingInPadding[$dir]=true# indicating that we have sth positive in this padding 
                #below we do actual dilatation
                if((($xMeta+1)+($xMetaChange)>0) && (($xMeta+1)+($xMetaChange))<=(metaDataDims[1])  && (($yMeta+1)+($yMetaChange)>0) && (($yMeta+1)+($yMetaChange)<=metaDataDims[2]) && (($zMeta+1)+($zMetaChange))>0 && (($zMeta+1)+($zMetaChange))<=metaDataDims[3]  )
                    @inbounds $mainArr[($xMeta)*dataBdim[1]+$a,($yMeta)*dataBdim[2]+$b,($zMeta)*dataBdim[3]+$c]=true
                    #checking is it marked as to be validates
                    if(areToBeValidated[$queueNumber])
                        #if we have true in reference array in analyzed spot
                        if($refArr[($xMeta)*dataBdim[1]+$a,($yMeta)*dataBdim[2]+$b,($zMeta)*dataBdim[3]+$c])
                            # aa = $a
                            # bb= $b
                            # cc = $c
                            # if($queueNumber==1)
                            #     CUDA.@cuprint "a $(aa) b $(bb) c $(cc) \n"    
                            # end        

                            #adding the result to the result list at correct spot - using metadata taken from metadata
                            @addResult(metaData
                            ,$xMeta+($xMetaChange)
                            ,$yMeta+($yMetaChange)
                            ,$zMeta+($zMetaChange)
                            ,resList
                            ,resListIndicies
                            ,(($xMeta)*dataBdim[1]+$a)
                            ,(($yMeta)*dataBdim[2]+$b)
                            ,(($zMeta)*dataBdim[3]+$c)
                            ,$dir
                            ,$iterNumb
                            ,$queueNumber
                            ,metaDataDims
                            ,mainArrDims
                            ,$isGold    )
                        end
                    end
                end
            end


    end)
    end)#quote
end


"""
In case we have any true in padding we need to set the next block as to be activated 
yet we need also to test wheather next block in this direction actually exists so we check metadata dimensions
metaData - multidim array with metadata about data blocks
isToBeActivated - boolean taken from res shmem indicating weather there was any  true in related direction
xMeta,yMeta,zMeta - current metadata spot 
isGold - indicating wheather we are in gold or other pass
xMetaChange,yMetaChange,zMetaChange - points in which direction we should go in analyzis of metadata - how to change current metaData 
metaDataDims - dimensions of metadata

"""
function setNextBlockAsIsToBeActivated(isToBeActivated::Bool,xMetaChange,yMetaChange,zMetaChange,xMeta,yMeta,zMeta,isGold,  metaData, metaDataDims)
   if(isToBeActivated && (xMeta)+xMetaChange<=metaDataDims[1]  && (yMeta)+yMetaChange>0 && (yMeta)+yMetaChange<=metaDataDims[2] && (zMeta)+zMetaChange>0 && (zMeta)+zMetaChange<=metaDataDims[3]) 
    metaData[(xMeta+1)+xMetaChange,(yMeta+1)+yMetaChange,(zMeta+1)+zMetaChange,getIsToBeActivatedInSegmNumb()-isGold  ]=1 
    end
end

"""
executes paddingProcessCombined over all paddings 
"""
macro processPadding(isGold,xMeta,yMeta,zMeta,iterNumb,mainArr,refArr)
    return esc(quote
         
    
         # #process top padding
        @paddingProcessCombined(loopAZFixed,loopBZfixed,dataBdim[1], dataBdim[2], x,y,1 ,0,0,-1,  $mainArr,$refArr,5,$iterNumb, 12-$isGold,$xMeta,$yMeta,$zMeta,$isGold)
       
        # #process bottom padding
        @paddingProcessCombined(loopAZFixed,loopBZfixed,dataBdim[1],dataBdim[2],x,y,dataBdim[3]+2,0,0,1,$mainArr,$refArr,6,$iterNumb,10-$isGold,$xMeta,$yMeta,$zMeta,$isGold)
        # 
    #process left padding
         @paddingProcessCombined(loopAXFixed,loopBXfixed,dataBdim[2], dataBdim[3],1,x,y , -1,0,0,  $mainArr,$refArr,1,$iterNumb, 4-$isGold,$xMeta,$yMeta,$zMeta,$isGold )
       
       
         # #process rigth padding
        @paddingProcessCombined(loopAXFixed,loopBXfixed,dataBdim[2], dataBdim[3], dataBdim[1]+2,x,y , 1,0,0,  $mainArr,$refArr,2,$iterNumb, 2-$isGold,$xMeta,$yMeta,$zMeta,$isGold)
       #process anterior padding
        @paddingProcessCombined(loopAYFixed,loopBYfixed,dataBdim[1], dataBdim[3], x,dataBdim[2]+2,y , 0,1,0,  $mainArr,$refArr,4,$iterNumb, 6-$isGold,$xMeta,$yMeta,$zMeta,$isGold)

        #process posterior padding
        @paddingProcessCombined(loopAYFixed,loopBYfixed,dataBdim[1], dataBdim[3], x,1,y , 0,-1,0,  $mainArr,$refArr,3,$iterNumb, 8-$isGold,$xMeta,$yMeta,$zMeta,$isGold)
        #checking res  shmem (we used part that is unused in  dilatation step) 
        
        sync_threads()
        

    end)

end#processPadding
           



end#ProcessMainDataVerB






# """
# we are processing padding 
# """
# macro processPadding()
#     return esc(quote
#     #so here we utilize iter3 with 1 dim fixed 
#     @unroll for  dim in 1:3, numb in [1,34]              
#         @iter3dFixed dim numb if( isPaddingToBeAnalyzed(resShmem,dim,numb ))
#                 if(val)#if value in padding associated with this spot is true
#                     # setting value in global memory
#                     @inbounds  analyzedArr[x,y,z]= true
#                     # if we are here we have some voxel that was false in a primary mask and is becoming now true - if it is additionaly true in reference we need to add it to result
#                     resShmem[1,1,1]=true
#                     if(@inbounds referenceArray[x,y,z])
#                         appendResultPadding(metaData, linIndex, x,y,z,iterationnumber, dim,numb)
#                 end#if
#                 #we need to check is next block in given direction exists
#                 if(isNextBlockExists(metaData,dim, numb ,linIter, isPassGold, maxX,maxY,maxZ))
#                     if(resShmem[1,1,1])
#                         @ifXY 1 1 setAsToBeActivated(metaData,linIndex,isPassGold)
#                     end    
#                 end # if isNextBlockExists       
#             end # if isPaddingToBeAnalyzed   
#         end#iter3dFixed       
#     end#for
# end)
# end





# """
# Help to establish should we validate the voxel - so if ok add to result set, update the main array etc
#   in case we have some true in padding
#   generally we need just to get idea if
#     we already had true in this very spot - if so we ignore it
#     can this spot be reached by other voxels from the block we are reaching into - in other words padding is analyzing the same data as other block is analyzing in its main part
#       hence if the block that is doing it in main part will reach this spot on its own we will ignore value from padding 

#   in order to reduce sears direction by 1 it would be also beneficial to know from where we had came - from what direction the block we are spilled into padding 
# """
# function isPaddingValToBeValidated(dir,analyzedArr, x,y,z )::Bool
     
# if(dir!=5)  if( @inbounds resShmem[threadIdxX(),threadIdxY(),zIter-1]) return false  end end #up
# if(dir!=6)  if( @inbounds  resShmem[threadIdxX(),threadIdxY(),zIter+1]) return false  end  end #down
    
# if(dir!=1)   if( @inbounds  resShmem[threadIdxX()-1,threadIdxY(),zIter]) return false  end  end #left
# if(dir!=2)   if( @inbounds   resShmem[threadIdxX()+1,threadIdxY(),zIter]) return false  end end  #right

# if(dir!=4)   if(  @inbounds  resShmem[threadIdxX(),threadIdxY()+1,zIter]) return false  end  end #front
# if(dir!=3)   if( @inbounds  resShmem[threadIdxX(),threadIdxY()-1,zIter]) return false  end end  #back
#   #will return true only in case there is nothing around 
#   return true
# end
