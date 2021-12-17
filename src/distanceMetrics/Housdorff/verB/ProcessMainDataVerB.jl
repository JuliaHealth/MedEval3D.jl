module ProcessMainDataVerB
using CUDA, Main.BitWiseUtils,Logging,Main.CUDAGpuUtils,Main.WorkQueueUtils, Logging,StaticArrays,Main.MetaDataUtils, Main.IterationUtils, Main.ReductionUtils, Main.CUDAAtomicUtils,Main.MetaDataUtils, Main.ResultListUtils
export @loadToshmemPaddings,@executeIterPadding,@loadToshmemPaddings,@validatePaddingInfo,@dilatateHelper,getDir,@validateData, @executeDataIterWithPadding, @loadMainValues,setNextBlockAsIsToBeActivated,@paddingProcessCombined,calculateLoopsIter,@processMaskData, @paddingIter,@processPadding


                
"""
 validates data is of our intrest               
"""                
macro validateData(isGold,xMeta,yMeta,zMeta,iterNumb,mainArr,refArr)
    return esc(quote
   #first we will load data from target arr so we can be sure that we are not ovewriting sth already written by diffrent thread block
   locArr = @inbounds($mainArr[($xMeta-1)*dataBdim[1]+ threadIdxX(),($yMeta-1)*dataBdim[2]+ threadIdxY(),$zMeta])
   # here we are anaylyzing only main part of the  data block paddings will be analyzed separately
   @unroll for bitIter in 1:32
       resShemVal = isBit1AtPos(@inbounds(shmemblockData[threadIdxX(),threadIdxY(),2]), bitIter)
       inSource = isBit1AtPos(locArr, bitIter)
       #later usefull to establish is mask full
       isMaskFull= (resShemVal && isMaskFull)
        # if($xMeta==3 && $yMeta==3 && $zMeta==3)
        #     CUDA.@cuprint " isMaskFull $(isMaskFull) resShemVal $(resShemVal) "
        # end    

    #    #so we have voxel that was not yet covered earlier, is covered now and is in refrence arr - so one where we establish is sth covered 
       if(resShemVal && !inSource)
        # we need reference array value to check weather we had covered anything new from other arrayin this iteration
            if((($refArr)[(($xMeta-1)*dataBdim[1]+ threadIdxX()),($yMeta-1)*dataBdim[2]+ threadIdxY(),($zMeta-1)*dataBdim[3]+ bitIter ])==numberToLooFor)
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
    #load data
    xm = $xMeta
    ym = $yMeta
    zm = $zMeta
    @inbounds shmemblockData[threadIdxX(),threadIdxY(),1] = $mainArr[(($xMeta-1)*dataBdim[1]+ threadIdxX()),(($yMeta-1)*dataBdim[2]+ threadIdxY()),$zMeta]
    #clear padding
    for iterY in 0:inBlockLoopXZIterWithPadding
        if((threadIdxY()+iterY*dataBdim[2])<=dataBdim[1]  )
            #we are reusing offsetIter
            offsetIter=0
            for bitPos in 2:7
                shmemPaddings[threadIdxX(),(threadIdxY()+iterY*dataBdim[2]),bitPos ]=false
            end
        end
    end             
    sync_threads()
    #now immidiately we can go with dilatation up and down and save it to res shmem we are not modyfing  locArr
    @inbounds shmemblockData[threadIdxX(),threadIdxY(),2]=@bitDilatate(shmemblockData[threadIdxX(),threadIdxY(),1])


    # idX = threadIdxX()
    # idY = (threadIdxY())
    # metX =    $xMeta
    # metY =   $yMeta
    # zm= $zMeta
    # x= (($xMeta-1)*dataBdim[1]+ threadIdxX())
    # y= (($yMeta-1)*dataBdim[2]+ threadIdxY())
    #  if(metX==5 && metY==5  && zm==3 && idX==1 && idY==2)
    # # if(metX==2 && metY==2  && zm==2)# && idX==2 && idY==2)
    #     CUDA.@cuprint "source to load  $(shmemblockData[threadIdxX(),threadIdxY(),1])  x $(x) y $(y) metX $(metX)  metY $(metY)  idX $(idX) idY$(idY) \n "
    # end   


    # now if we have values in first or last bit we need to modify appropriate spots in the shmemPaddings
    locBool = isBit1AtPos(shmemblockData[threadIdxX(),threadIdxY(),1],dataBdim[3])#bottom
    if(locBool)
        isAnythingInPadding[2]= true 
    end    
    @inbounds shmemPaddings[threadIdxX(),threadIdxY(),2]=locBool

    locBool = isBit1AtPos(shmemblockData[threadIdxX(),threadIdxY(),1],1)#top
    @inbounds shmemPaddings[threadIdxX(),threadIdxY(),7]= locBool   
    if(locBool)
        isAnythingInPadding[7]= true 
    end   

    #now we will  do left - right dilatations howvewer we must be sure that we checked boundary conditions 
    sync_threads()
    #left
    @dilatateHelper((threadIdxX()==1), 3,bitPos,threadIdxY(),(-1), (0))

    #right
    @dilatateHelper((threadIdxX()==dataBdim[1]), 4,bitPos,threadIdxY(),(1), (0))

    #  #posterior
    @dilatateHelper((threadIdxY()==1), 6,threadIdxX(), bitPos,(0), (-1))

      #anterior 
    @dilatateHelper((threadIdxY()==dataBdim[2]), 5,threadIdxX(), bitPos,(0), (1))

    sync_threads()
    #now we need to persist the paddings still becouse its size is up to 32 by 32 we need to iterate over y dimension
    for iterY in 0:inBlockLoopXZIterWithPadding
        if((threadIdxY()+iterY*dataBdim[2])<=dataBdim[1]  )
            #we are reusing offsetIter
            offsetIter=0
            for bitPos in 2:7

                # idX = threadIdxX()
                # idY = (threadIdxY()+iterY*dataBdim[2])
                # if(idX==1 && idY==1)
                #     CUDA.@cuprint "bittt $(bitPos)  isTrue    $(shmemPaddings[threadIdxX(),(threadIdxY()+iterY*dataBdim[2]),bitPos ]) \n"
                # end    


                offsetIter=  @setBitTo(offsetIter,bitPos, shmemPaddings[threadIdxX(),(threadIdxY()+iterY*dataBdim[2]),bitPos ])
            end

            @inbounds paddingStore[$xMeta,$yMeta,$zMeta,threadIdxX(),(threadIdxY()+iterY*dataBdim[2])]= offsetIter
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
    # locBool = false


  
    if($predicate)
        for bitPos in 1:32

                shmemPaddings[$padingVariedA,$padingVariedB,$paddingPos]=isBit1AtPos(shmemblockData[threadIdxX(),threadIdxY(),1],bitPos)
                #we need to mark weather there is anything in padding so we will mark block as to be activates 
                if(shmemPaddings[$padingVariedA,$padingVariedB,$paddingPos]) 
                    # if(metX==2 && metY==2 && idX==1 && idY==1 && zm==2)
                    #     padvarA = $padingVariedA
                    #     padVarB = $padingVariedB
                    #     padPos = $paddingPos
                    #     CUDA.@cuprint "shmem pos 2 padvarA $(padvarA)  padVarB $(padVarB) padPos $(padPos) \n"   
                    # end   
                    
                    isAnythingInPadding[$paddingPos]= true 
                end
        end#for

    else
           

        shmemblockData[threadIdxX(),threadIdxY(),2] = @bitPassOnes(shmemblockData[threadIdxX(),threadIdxY(),2],shmemblockData[threadIdxX()+($normalXChange),threadIdxY()+($normalYchange),1]   )

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
    return if((bitIter-1)>0 && isBit1AtPos(@inbounds(shmemblockData[threadIdxX(),threadIdxY(),1]), bitIter-1) ) 
                6
            elseif(((bitIter)<dataBdim[3]) && isBit1AtPos(@inbounds(shmemblockData[threadIdxX(),threadIdxY(),1]), bitIter+1) ) 
                5
            elseif((threadIdxX()-1>0) && isBit1AtPos(@inbounds(shmemblockData[threadIdxX()-1,threadIdxY(),1]), bitIter) )
                2
            elseif((threadIdxX()<dataBdim[1]) && isBit1AtPos(@inbounds(shmemblockData[threadIdxX()+1,threadIdxY(),1]), bitIter) )
                1
            elseif((threadIdxY()-1>0) && isBit1AtPos(@inbounds(shmemblockData[threadIdxX(),threadIdxY()+1,1]), bitIter) )
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
  @ifY 1 if(threadIdxX()<15) areToBeValidated[threadIdxX()] =(metaData[($xMeta),($yMeta),($zMeta),(getIsToBeAnalyzedNumb() +threadIdxX())] ==1 )end 
  @loadMainValues($mainArr,$xMeta,$yMeta,$zMeta)                                       
  sync_threads()
  ########## check data aprat from padding
  #can be skipped if we have the block with already all results analyzed 
  isMaskFull= true
  if(areToBeValidated[14-$isGold])
      @validateData($isGold,$xMeta,$yMeta,$zMeta,$iterNumb,$mainArr,$refArr) 
  end 
  #save all dilatations apart from padding ...                 
  $mainArr[(($xMeta-1)*dataBdim[1]+ threadIdxX()),(($yMeta-1)*dataBdim[2]+ threadIdxY()),$zMeta] = @inbounds(shmemblockData[threadIdxX(),threadIdxY(),2])
  #now we need to establish are we full here; and whether neighbours are to be activated
  isMaskFull =syncThreadsAnd(isMaskFull)
  @ifXY 1 1 setNextBlockAsIsToBeActivated(isAnythingInPadding[3] ,-1,0,0,$xMeta,$yMeta,$zMeta,$isGold,  metaData, metaDataDims)
  @ifXY 2 1 setNextBlockAsIsToBeActivated(isAnythingInPadding[4] ,1,0,0,$xMeta,$yMeta,$zMeta,$isGold,  metaData, metaDataDims)
  @ifXY 3 1 setNextBlockAsIsToBeActivated(isAnythingInPadding[6] ,0,-1,0,$xMeta,$yMeta,$zMeta,$isGold,  metaData, metaDataDims)
  @ifXY 4 1 setNextBlockAsIsToBeActivated(isAnythingInPadding[5] ,0,1,0,$xMeta,$yMeta,$zMeta,$isGold,  metaData, metaDataDims)
  @ifXY 5 1 setNextBlockAsIsToBeActivated(isAnythingInPadding[2] ,0,0,-1,$xMeta,$yMeta,$zMeta,$isGold,  metaData, metaDataDims)
  @ifXY 6 1 setNextBlockAsIsToBeActivated(isAnythingInPadding[7] , 0,0,1,$xMeta,$yMeta,$zMeta,$isGold,  metaData, metaDataDims)
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

    #by construction one thread will neeed to load just one integer into its registers and to res
    @inbounds shmemblockData[threadIdxX(),threadIdxY(),1] = $mainArr[($xMeta-1)*dataBdim[1]+ threadIdxX(),($yMeta-1)*dataBdim[2]+ threadIdxY(),$zMeta]
    # we already gone through dilatation and metadata analysis so now we are analysing paddings from previous loop
    #first we need to load data from paddings of surrounding blocks and  push it into shared memory
    #yet we need to check weather in given direction there is some block so if we are on edge ?
    #order of paddings for reference 1)top, 2)bottom, 3)left 4)right , 5)anterior, 6)posterior
    
    #for example here we look to the right  block so we are putting data in our local right padding from left padding of neighpbouring block
    #right
    @loadToshmemPaddings($xMeta,$yMeta,$zMeta, 1,0,0, 3, 4)
    #left
    @loadToshmemPaddings($xMeta,$yMeta,$zMeta, (-1),0,0, 4, 3)
    #anterior
    @loadToshmemPaddings($xMeta,$yMeta,$zMeta, 0,1,0, 6, 5)
    #posterior
    @loadToshmemPaddings($xMeta,$yMeta,$zMeta, 0,(-1),0, 5, 6)
    #top
    @loadToshmemPaddings($xMeta,$yMeta,$zMeta, 0,0,1, 7, 2)
    #bottom
    @loadToshmemPaddings($xMeta,$yMeta,$zMeta, 0,0,(-1), 2, 7)
    #so now we have data loaded from surrounding blocks about their paddings we may now modify accordingly current block

    sync_threads()
    # now we have data loaded about what is already in dilatation array what was marked from paddings of neighbouring blocks
    # so we need to add dilatation and if needed result to result list 
    
    #top
    @validatePaddingInfo(shmemPaddings[threadIdxX(),threadIdxY(),7]#shmemVal
    ,threadIdxX()#xpos
    ,threadIdxY()#ypos
    ,1#zpos
    ,$isGold,
         6#dir
         ,7#shmemPaddingLayer 
         ,$refArr,$xMeta,$yMeta,$zMeta,$iterNumb
          )    
    #bottom
    @validatePaddingInfo(shmemPaddings[threadIdxX(),threadIdxY(),2]#shmemVal
    ,threadIdxX()#xpos
    ,threadIdxY()#ypos
    ,dataBdim[3]#zpos
    ,$isGold,
         5#dir
         ,2#shmemPaddingLayer 
         ,$refArr,$xMeta,$yMeta,$zMeta,$iterNumb
          )
    sync_threads()      
    #now anterior, posterior, left and right need to be evaluated only on one 
    @unroll for bitPos in 1:32     
        #left
       @ifY 1 if(threadIdxX()<=dataBdim[2]) 
        @validatePaddingInfo(shmemPaddings[bitPos,threadIdxX(),3]#shmemVal
        ,1#xpos
        ,threadIdxX()#ypos
        ,bitPos#zpos
        ,$isGold,
            2#dir
            ,3#shmemPaddingLayer 
            ,$refArr,$xMeta,$yMeta,$zMeta,$iterNumb
            )
       end
        #right
        @ifY 2 if(threadIdxX()<=dataBdim[2]) 
             @validatePaddingInfo(shmemPaddings[bitPos,threadIdxX(),4]#shmemVal
            ,dataBdim[1]#xpos
            ,threadIdxX()#ypos
            ,bitPos#zpos
            ,$isGold,
                1#dir
                ,4#shmemPaddingLayer 
                ,$refArr,$xMeta,$yMeta,$zMeta,$iterNumb
                )
           end


        #anterior
        @ifY 3 begin @validatePaddingInfo(
            shmemPaddings[threadIdxX(),bitPos,5]#shmemVal
            ,threadIdxX()#xpos
            ,dataBdim[2]#bitPos#ypos
            ,bitPos  #zpos
            ,$isGold,
                3#dir
                ,5#shmemPaddingLayer 
                ,$refArr,$xMeta,$yMeta,$zMeta,$iterNumb
                )
           end
        #posterior
        @ifY 4 begin @validatePaddingInfo(shmemPaddings[threadIdxX(),bitPos,6]#shmemVal
            ,threadIdxX()#xpos
            ,1#ypos
            ,bitPos#zpos
            ,$isGold,
                4#dir
                ,6#shmemPaddingLayer 
                ,$refArr ,$xMeta,$yMeta,$zMeta,$iterNumb           )
           end

    end#for bit pos     
    sync_threads()


    $mainArr[(($xMeta-1)*dataBdim[1]+ threadIdxX()),(($yMeta-1)*dataBdim[2]+ threadIdxY()),($zMeta)] =@inbounds(shmemblockData[threadIdxX(),threadIdxY(),1])
    
end)
end#executeIterPadding



"""
helper macro for executeIterPadding - if the predicate is true (predicate check given spot in the shmemPaddings ) function executes
xpos,ypos,zpos - current position in data block
dir - direction from which we got this so for example in top padding we got it dilatated from top
    top 6 
    bottom 5  
    left 2
    right 1 
    anterior 3
    posterior 4
 shmemVal -  boolean padding value of intrest  
 shmemPaddingLayer - which layer it is reference below
    1)top, 2)bottom, 3)left 4)right , 5)anterior, 6)posterior 
"""
macro validatePaddingInfo(shmemVal,xpos,ypos,zpos,isGold, dir,shmemPaddingLayer,refArr,xMeta,yMeta,zMeta,iterNumb )
    return esc(quote
    shmm = $shmemVal
    xp = (($xMeta-1)*dataBdim[1]+ $xpos)
    yp = (($yMeta-1)*dataBdim[2]+ $ypos)
    zp = (($zMeta-1)*dataBdim[3]+ $zpos)
    bitPP = $zpos
    xPP = threadIdxX()
    smhmemPadLay = $shmemPaddingLayer
    if(($xMeta)==6 && ($yMeta)==5  && ($zMeta)==6)  
       #@ifXY 1 1  CUDA.@cuprint  "aaa  $(shmemPaddings[5,2,4])   " 
        CUDA.@cuprint "in validate bitPP $(bitPP) xPP $(xPP) shmm $(shmm) smhmemPadLay $(smhmemPadLay)\n"
    end  

    if($shmemVal)

        @setBitTo1(shmemblockData[$xpos,$ypos,1],$zpos)


        # CUDA.@cuprint "set bit to x $(xp)   \n"
        # if(($xMeta)>3)
        #     CUDA.@cuprint " pre reff in validate xp $(xp) yp $(yp) zp $(zp) bitPP $(bitPP) xPP $(xPP) shmm $(shmm) smhmemPadLay $(smhmemPadLay)\n"
        # end    
       
        if($refArr[(($xMeta-1)*dataBdim[1]+ $xpos)
            ,(($yMeta-1)*dataBdim[2]+ $ypos)
            ,(($zMeta-1)*dataBdim[3]+ $zpos)] 
            == numberToLooFor)
            # if(($xMeta)>3)
            #     CUDA.@cuprint "post reff  in validate bitPP $(bitPP) xPP $(xPP) shmm $(shmm) smhmemPadLay $(smhmemPadLay)\n"
            # end    
            @addResult(metaData
            ,$xMeta,$yMeta ,$zMeta,resList
            ,(($xMeta-1)*dataBdim[1]+ $xpos)
            ,(($yMeta-1)*dataBdim[2]+ $ypos)
            ,(($zMeta-1)*dataBdim[3]+ $zpos)
            ,$dir
            ,$iterNumb
            ,getIndexOfQueue($xpos,$ypos,$zpos,dataBdim,(1-$isGold))
            ,metaDataDims
            ,mainArrDims
            ,$isGold)
        end#if is in referenca array    
    end#if true in padding    
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
    if( ((($xMeta)+$xMetaChange)<=metaDataDims[1])
         && (($xMeta)+$xMetaChange>0)
        && (($yMeta)+$yMetaChange>0)
         && ((($yMeta)+$yMetaChange)<=metaDataDims[2])
         && (($zMeta)+$zMetaChange>0)
         && ((($zMeta)+$zMetaChange)<=metaDataDims[3])
         )
        @unroll for iterY in 0:inBlockLoopXZIterWithPadding
            if((threadIdxY()+iterY*dataBdim[2])<=dataBdim[1]  )
                #so we are intrested only in given bit from neighbouring block
                booll = isBit1AtPos(paddingStore[($xMeta+$xMetaChange),($yMeta+$yMetaChange),($zMeta+$zMetaChange),threadIdxX(),(threadIdxY()+iterY*dataBdim[2])],($bitOfIntrest))
                #     xm = (($xMeta)+$xMetaChange)
                #     ym = ($yMeta)+$yMetaChange
                #     zm = ($zMeta)+$zMetaChange
                #     targetNumb = $shmemPaddingTargetNumb
                #     numb =(Int64(paddingStore[($xMeta+$xMetaChange),($yMeta+$yMetaChange),($zMeta+$zMetaChange),threadIdxX(),(threadIdxY()+iterY*dataBdim[2])]))
                # #if(($xMeta)==5 && ($yMeta)==5  && ($zMeta)==2  && threadIdxX()==4 && (threadIdxY()+iterY*dataBdim[2])==4)  #&& zm==1
                # # if(xm==5 && ym==5  && booll)  #&& zm==1
                # if(targetNumb==5 && ($xMeta)==5 && (($yMeta)==4  )  &&  (($zMeta)==5 ) && numb>0)  
                #     bitt = $bitOfIntrest
                #     CUDA.@cuprint "in load xm $(xm) ym $(ym) zm $(zm) idx $(threadIdxX())  idY $(threadIdxY()+iterY*dataBdim[2])  layer $(bitt) bool $(booll) targetNumb $(targetNumb) numb $(numb)) \n"
                # end                   
                shmemPaddings[threadIdxX(),(threadIdxY()+iterY*dataBdim[2]),$shmemPaddingTargetNumb ]=booll
            end    
        end#for    
    end#if such block exists    
    end)
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
   if(isToBeActivated && (xMeta)+xMetaChange<=metaDataDims[1]  && (xMeta)+xMetaChange>0 &&(yMeta)+yMetaChange>0 && (yMeta)+yMetaChange<=metaDataDims[2] && (zMeta)+zMetaChange>0 && (zMeta)+zMetaChange<=metaDataDims[3]) 
    metaData[(xMeta+1)+xMetaChange,(yMeta+1)+yMetaChange,(zMeta+1)+zMetaChange,getIsToBeActivatedInSegmNumb()-isGold  ]=1 
    end
end

           



end#ProcessMainDataVerB

