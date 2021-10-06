

"""
loads and do the main processing of data in arrays of intrest (padding of shmem will be analyzed separately)

"""
module ProcessMainData
using  StaticArrays,Main.CUDAGpuUtils ,Main.HFUtils, CUDA, Main.ProcessPadding
export executeDataIterFirstPass,executeDataIterOtherPasses,processMaskData,executeDataIterFirstPassWithPadding


# """
# loads and do the main processing of data in arrays of intrest (padding of shmem will be analyzed separately)
# analyzedArr - array we are currently dilatating
# refAray -array we are referencing (we do not dilatate it only check against it )
# iterationNumber - at what iteration of dilatation we are - so how many dilatations we already performed
# blockBeginingX,blockBeginingY,blockBeginingZ - coordinates where our block is begining - will be used as offset by our threads
# isMaskFull,isMaskEmpty - enables later checking is mask is empty full or neither
# resShmem - shared memory bit array of the same dimensions as data block but plus 2 
# sourceShmem - bit array of the same dimensions as data block 
# locArr - local bit array of thread
# resArray- 3 dimensional array where we put results
# loopX,loopY,loopZ - how many times main loops need to be invoked in order for the thread block to cover all of  data block
# dataBlockDims - size of data block - so max x y and z possible index in those dimensions
# """
# function executeDataIterFirstPass(analyzedArr, referenceArray,blockBeginingX,blockBeginingY,blockBeginingZ,resShmem,sourceShmem,resArray,resArraysCounter, loopX,loopY,loopZ, dataBlockDims)
#     locArr = Int32(0)
#     isMaskFull= true
#     isMaskEmpty = true
#     #locArr.x |= true << UInt8(2)

#     ############## upload data
#     ---- to iteration3d we can get a function from macro to a generalized function and then produce multiple macros that will be chosen  based on multiple dispatch
#          same will get the macro iter3d with val so value will also be available  also we need to   we can also specify bounds safe and not safe variant - what will reduce number of ifs
#                     - also we can consider 
#     ###step 1            
#     @iter3dWithVal  dataBlockDims loopX loopY loopZ blockBeginingX blockBeginingY blockBeginingZ analyzedArr begin
#         #val is given by macro as value of this x,y,z 
#         locArr|= val << (zIter-1)
#         processMaskData( val, zIter, resShmem) 
#         #zIter given in macro as we are iterating in this spot
#         sourceShmem[threadIdxX(), threadIdxY(), zIter]                
#     end
#     syncthreads()
                    
#     ##step 2  
#     ########## check data aprat from padding
#     @iter3dWithVal  dataBlockDims loopX loopY loopZ blockBeginingX blockBeginingY blockBeginingZ analyzedArr begin
#         locVal::Bool = @inbounds  (locArr>>(zIter-1) & 1)
#         resShemVal::Bool = @inbounds resShmem[threadIdxX()+1,threadIdxY()+1,zIter+1]             
#     end
                    
   
#     ########## check data aprat from padding
#     @unroll for zIter::UInt8 in UInt8(1):UInt8(32)# most outer loop is responsible for z dimension
#         locVal::Bool = @inbounds  (locArr>>(zIter-1) & 1)
#         shmemVal::Bool = @inbounds resShmem[threadIdxX()+1,threadIdxY()+1,zIter+1]
#         # CUDA.@cuprint "locVal $(locVal)  shmemVal $(shmemVal) \n  "
#         locValOrShmem = (locVal | shmemVal)
#         isMaskFull= locValOrShmem & isMaskFull
#         isMaskEmpty = ~locValOrShmem & isMaskEmpty

#         #CUDA.@cuprint "locVal $(locVal)  shmemVal $(shmemVal) \n  "
#         if(!locVal && shmemVal)
#             # setting value in global memory
#             @inbounds  analyzedArr[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]= true
#             # if we are here we have some voxel that was false in a primary mask and is becoming now true - if it is additionaly true in reference we need to add it to result
#             isInReferencaArr::Bool= @inbounds referenceArray[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]
#             if(isInReferencaArr)
#                 #CUDA.@cuprint "isInReferencaArr $(isInReferencaArr) \n  "
#                 @inbounds  resArray[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]=UInt16(1)
#                 #CUDA.atomic_inc!(pointer(resArraysCounter), Int32(1))              
#                 atomicallyAddOneInt(resArraysCounter)
#             end#if
#         end#if
#       end#for

# end#executeDataIter




"""
   loads main values from analyzed array into shared memory and to locArr - which live in registers             
"""                
                
macro loadMainValues()
        @iter3dWithVal  dataBlockDims loopX loopY loopZ blockBeginingX blockBeginingY blockBeginingZ analyzedArr begin
        #val is given by macro as value of this x,y,z 
        locArr|= val << (zIter-1)
        processMaskData( val, zIter, resShmem) 
        #zIter given in macro as we are iterating in this spot
        sourceShmem[threadIdxX(), threadIdxY(), zIter]                
    end                
                end #loadMainValues
                
                
"""
 validates data is of our intrest               
"""                
macro validateData()
    @iter3dW  dataBlockDims loopX loopY loopZ blockBeginingX blockBeginingY blockBeginingZ resShemVal begin
        locVal::Bool = @inbounds  (locArr>>(zIter-1) & 1)
        resShemVal::Bool = @inbounds resShmem[threadIdxX()+1,threadIdxY()+1,zIter+1]             
        locValOrShmem = (locVal | resShemVal)
        #those needed to establish weather data block will remain active
        isMaskFull= locValOrShmem & isMaskFull
        @ifverr zzz isMaskEmpty = ~locValOrShmem & isMaskEmpty
        if(!locVal && resShemVal)       
              innerValidate(analyzedArr,referenceArray,x,y,z,privateResArray,privateResCounter,iterationnumber,sourceShmem  )
        end#if
     end#3d iter 
    
    
 end  #validateData                  

"""
this will be invoked when we know that we have a true in a spot that was false before this dilatation step and its task is to set to true appropriate spot in global array
- so proper dilatation
check weather we have true also in reference array - if so we  need to add this spot to the block result list in case we are invoke it from padding we need to look even futher into the
next block data to establish could this spot be activated from there
"""
  function innerValidate(analyzedArr,referenceArray,x,y,z,privateResArray,privateResCounter,iterationnumber,sourceShmem  )
            # setting value in global memory
            @inbounds  analyzedArr[x,y,z]= true
            # if we are here we have some voxel that was false in a primary mask and is becoming now true - if it is additionaly true in reference we need to add it to result
    
            if(@inbounds referenceArray[x,y,z])
                #results now are stored in a matrix where first 3 entries are x,y,z coordinates entry 4 is in which iteration we covered it and entry 5 from which direction - this will be used if needed        
                #privateResCounter privateResArray are holding in metadata blocks results and counter how many results were already added 
                #in each thread block we will have separate rescounter, and res array for goldboolpass and other pass
               direction=  @ifverr zzz  getDir(sourceShmem) | 0    
               @append  privateResArray privateResCounter  [x,y,z,iterationnumber, direction]      

            end#if
  end#innerValidate 
     


"""
Help to establish should we validate the voxel - so if ok add to result set, update the main array etc
  in case we have some true in padding
  generally we need just to get idea if
    we already had true in this very spot - if so we ignore it
    can this spot be reached by other voxels from the block we are reaching into - in other words padding is analyzing the same data as other block is analyzing in its main part
      hence if the block that is doing it in main part will reach this spot on its own we will ignore value from padding 

  in order to reduce sears direction by 1 it would be also beneficial to know from where we had came - from what direction the block we are spilled into padding 
"""
function isPaddingValToBeValidated(dir,analyzedArr, x,y,z )::Bool
     
if(dir!=5)  if( @inbounds resShmem[threadIdxX(),threadIdxY(),zIter-1]) return false  end end #up
if(dir!=6)  if( @inbounds  resShmem[threadIdxX(),threadIdxY(),zIter+1]) return false  end  end #down
    
if(dir!=1)   if( @inbounds  resShmem[threadIdxX()-1,threadIdxY(),zIter]) return false  end  end #left
if(dir!=2)   if( @inbounds   resShmem[threadIdxX()+1,threadIdxY(),zIter]) return false  end end  #right

if(dir!=4)   if(  @inbounds  resShmem[threadIdxX(),threadIdxY()+1,zIter]) return false  end  end #front
if(dir!=3)   if( @inbounds  resShmem[threadIdxX(),threadIdxY()-1,zIter]) return false  end end  #back
  #will return true only in case there is nothing around 
  return true
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
   function getDir(sourceShmem)
    if( @inbounds resShmem[threadIdxX(),threadIdxY(),zIter-1]) return 6  end  #up
    if( @inbounds  resShmem[threadIdxX(),threadIdxY(),zIter+1]) return 5  end #down
    
    if( @inbounds  resShmem[threadIdxX()-1,threadIdxY(),zIter]) return 2  end #left
    if( @inbounds   resShmem[threadIdxX()+1,threadIdxY(),zIter]) return 1  end #right

    if(  @inbounds  resShmem[threadIdxX(),threadIdxY()+1,zIter]) return 3  end #front
     if( @inbounds  resShmem[threadIdxX(),threadIdxY()-1,zIter]) return 4  end #back
  end#getDir
    
    
####to 3d iter data 
      ---- to iteration3d we can get a function from macro to a generalized function and then produce multiple macros that will be chosen  based on multiple dispatch
         same will get the macro iter3d with val so value will also be available  also we need to   we can also specify bounds safe and not safe variant - what will reduce number of ifs
                    - also we can consider 
    ----------- need to add in 3 dim iter a possibility to customize the way how we define offsets in x,y,z so instead of grid or block dim we need to have ability to do it separately as extra arguments
ifverr - will return only this expression that is compatible with version number supplied
###                                    

privateResCounter, blockMaxRes - 
function executeDataIterFirstPassWithPadding(analyzedArr, referenceArray,blockBeginingX
                                ,blockBeginingY,blockBeginingZ,resShmem,sourceShmem,resArray,resArraysCounter
                                ,currBlockX,currBlockY,currBlockZ,isPassGold,metaData,metadataDims
                                ,mainQuesCounter,mainWorkQueue,iterationNumber,debugArr, loopX,loopY,loopZ, dataBlockDims
                                privateResCounter, blockResCounter)
    
    
    locArr::UInt32 = UInt32(0)
    isMaskFull::Bool= true
    isMaskEmpty::Bool = true
    isMaskOkForProcessing::Bool = true
    offset = UInt8(1)
    # nextBlockX::UInt8 = UInt8(0)
    # nextBlockY::UInt8 =UInt8(0)
    # nextBlockZ::UInt8 = UInt8(0)
    #locArr.x |= true << UInt8(2)

    ############## upload data
  ############## upload data
        ###step 1            
@loadMainValues
                                        
    syncthreads()
        ---------  can be skipped if we have the block with already all results analyzed - we know it from block private counter
if(privateResCounter[1]<blockMaxRes[1])
   @validateData 
end                  
    ##step 2  
    ########## check data aprat from padding
  
     ################################################################################################################################ 
     #processing padding



 
     ################################################################################################################################ 


    #function will be invoked on paddings from all sides
    # function innerProcessPadding(   resShmemX::UInt8 # x value to find padding value of intrest in shared memory
    #                                 ,resShmemY::UInt8 # y value to find padding value of intrest in shared memory
    #                                 ,resShmemZ::UInt8 # z value to find padding value of intrest in shared memory
    #                                 ,primaryZiter::UInt8 # the same as we analyzed above what blocks should be analyzed
    #                                 ,correctedX::UInt16 # x value to find padding value of intrest in global memory
    #                                 ,correctedY::UInt16# y value to find padding value of intrest in global memory
    #                                 ,correctedZ::UInt16)# z value to find padding value of intrest in global memory
    resShmemX=threadIdxX()#resShmemX
    resShmemY=threadIdxY()#resShmemY
    resShmemZ=1#resShmemZ
    primaryZiter=6 #primaryZiter
    correctedX=blockBeginingX+threadIdxX()#correctedX
    correctedY=blockBeginingY +threadIdxY()#correctedY
    correctedZ=blockBeginingZ-1#correctedZ
     

    if(resShmem[2,primaryZiter+1,2] && resShmem[primaryZiter+1,2,3] )        
        #we check for each lane is there a true in respective spot
        if(resShmem[resShmemX,resShmemY,resShmemZ])
            locArr= atomicallyAddOneInt(mainQuesCounter)+1  
            
            # when we set new result from padding we need to take into account possibility that neighbour block already did it
            # hence when we set atomic we check the old value if old value was 0 all is good if not we  do not increse  the rescounter       
   ----------- now we have block private results so this  need to be utilized on the neighbouring bblocks still  we need to keep data races consideration living
            if(resArray[correctedX,correctedY,correctedZ]==0)
               resArray[correctedX,correctedY,correctedZ]=UInt16(iterationNumber)
               atomicallyAddOneInt(resArraysCounter)
            end 
 
            #updating work queue
            @inbounds mainWorkQueue[locArr,1]= (currBlockX+(primaryZiter==1)-(primaryZiter==2)) #x,y,z dim of block in metadata
            @inbounds mainWorkQueue[locArr,2]=currBlockY+(primaryZiter==3)-(primaryZiter==4) #x,y,z dim of block in metadata
            @inbounds mainWorkQueue[locArr,3]=currBlockZ+(primaryZiter==5)-(primaryZiter==6) #x,y,z dim of block in metadata
            @inbounds mainWorkQueue[locArr,4]=UInt8(isPassGold) #x,y,z dim of block in metadata

        end #if          
          

    end#if

        #end#innerProcessPadding
    
    ######################################## 
    #now we will analyze the padding planes one by one using innerProcessPadding


    # if(UInt8(threadIdxY())==zIter   &&  (threadIdxX()==2) )
    #     debugArr[zIter] = resShmem[zIter+1,2,3]  
    # end   


                  
    ######## TOP
    # innerProcessPadding(threadIdxX()#resShmemX
    #                     ,threadIdxY()#resShmemY
    #                     ,1#resShmemZ
    #                     ,6 #primaryZiter
    #                     ,blockBeginingX+threadIdxX()#correctedX
    #                     ,blockBeginingY +threadIdxY()#correctedY
    #                     ,blockBeginingZ-1#correctedZ
    #                     )
   
#    ######## BOTTOM
#    innerProcessPadding(threadIdxX()#resShmemX
#                         ,threadIdxY()#resShmemY
#                         ,34#resShmemZ
#                         ,5 #primaryZiter
#                         ,blockBeginingX+threadIdxX()#correctedX
#                         ,blockBeginingY +threadIdxY()#correctedY
#                         ,blockBeginingZ+33#correctedZ
#                         )
                    

#    ######## LEFT
#    innerProcessPadding(1#resShmemX
#                         ,threadIdxX()#resShmemY
#                         ,threadIdxY()#resShmemZ
#                         ,2 #primaryZiter
#                         ,blockBeginingX-1#correctedX
#                         ,blockBeginingY +threadIdxX()#correctedY
#                         ,blockBeginingZ+threadIdxY()#correctedZ
#                         )
#    ######## RIGHT
#    innerProcessPadding(1#resShmemX
#                         ,threadIdxX()#resShmemY
#                         ,threadIdxY()#resShmemZ
#                         ,1 #primaryZiter
#                         ,blockBeginingX+33#correctedX
#                         ,blockBeginingY +threadIdxX()#correctedY
#                         ,blockBeginingZ+threadIdxY()#correctedZ
#                         )                        
#    ######## Anterior
#    innerProcessPadding(threadIdxX()#resShmemX
#                         ,34#resShmemY
#                         ,threadIdxY()#resShmemZ
#                         ,3 #primaryZiter
#                         ,blockBeginingX+threadIdxX()#correctedX
#                         ,blockBeginingY +33#correctedY
#                         ,blockBeginingZ+threadIdxY()#correctedZ
#                         )   
#    ######## Posterior
#    innerProcessPadding(threadIdxX()#resShmemX
#                         ,1#resShmemY
#                         ,threadIdxY()#resShmemZ
#                         ,4 #primaryZiter
#                         ,blockBeginingX+threadIdxX()#correctedX
#                         ,blockBeginingY -1#correctedY
#                         ,blockBeginingZ+threadIdxY()#correctedZ
#                         )   
sync_threads()

###########################################
#after all processing we need to establish weather the mask is full  (in case of first iteration also is it empty)
    function isMaskStillActive()




   ----------- this can be done nearly completely be reduction macros also in order to reduce number of reductions we can consider fusing it with step above
    # #reduction
    #         offset = UInt16(1)
    #         while(offset <32) 
    #             isMaskFull = isMaskFull & shfl_down_sync(FULL_MASK, isMaskFull, offset)  
    #             isMaskEmpty = isMaskEmpty & shfl_down_sync(FULL_MASK, isMaskEmpty, offset)  
    #             offset<<= 1
    #         end
    #         @ifX 1  begin 
    #             @inbounds  resShmem[2,threadIdxY()+1,6]=isMaskFull
    #             @inbounds  resShmem[2,threadIdxY()+1,7]=isMaskEmpty
    #         end   

    #     sync_threads()#now we have partially reduced values marking wheather we have full or empty block
    #     # we get full reductions
    #     offset = UInt16(1)
    #     if(threadIdxY()==1)
    #         while(offset <32)
    #             @inbounds  resShmem[2,threadIdxX()+1,6] = shfl_down_sync(FULL_MASK,resShmem[2,threadIdxX()+1,6], offset)
    #             @inbounds  resShmem[2,threadIdxX()+1,7] = shfl_down_sync(FULL_MASK,resShmem[2,threadIdxX()+1,7], offset)
    #         end#while    
    #         end#if 
    #     #now we have in resShmem[2,2,6] boolean marking is data block is full of trues and in resShmem[2,2,7] if there is no true at all
    #     ##########  now we need to update meta data in case we have now empty or full block
    #     if(threadIdxY()==5 && threadIdxX()==5 && (resShmem[2,2,6] || resShmem[2,2,7]))
    #         metaData[currBlockX,currBlockY,currBlockZ,isPassGold+1]=false # we set is inactive 
    #     end#if   
    #     if(threadIdxY()==6 && threadIdxX()==6 && (resShmem[2,2,6] || resShmem[2,2,7]))
    #         metaData[currBlockX,currBlockY,currBlockZ,isPassGold+3]=true # we set is as full
    #     end#if
    #     #so in case it not empty and not full we need to put it into the work queue and increment appropriate counter
    #     if(threadIdxY()==7&& threadIdxX()==7 && !(resShmem[2,2,6] || resShmem[2,2,7]))
    #         mainWorkQueue[atomicallyAddOneInt(mainQuesCounter)+1]=[currBlockX,currBlockY,currBlockZ,UInt8(isPassGold)]    
    #     end#if
    #     #######now in case block remains neither full nor empty we add it to the work queue






end#isMaskStillActive
sync_threads()

end#executeDataIter











# """
# specializes executeDataIterFirstPass as it do not consider possibility of block being empty 
# """
# function executeDataIterOtherPasses(analyzedArr, refAray,iterationNumber ,blockBeginingX,blockBeginingY,blockBeginingZ,isMaskFull,resShmem,locArr,resArray,resArraysCounter)
#     @unroll for zIter in UInt16(1):32# most outer loop is responsible for z dimension
#         local locBool = testArrInn[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]
#         locArr|= locBool << zIter
#        processMaskData( locBool, zIter, resShmem)
#        CUDA.unsafe_free!(locBool)
#     end#for 
#     sync_threads() #we should have in resShmem what we need 
#     @unroll for zIter in UInt16(1):32 # most outer loop is responsible for z dimension - importnant in this loop we ignore padding we will deal with it separately
#          local locBoolRegister = (locArr>>zIter & UInt32(1))==1
#          local locBoolShmem = resShmem[threadIdxX()+1,threadIdxY()+1,zIter+1]
#          validataDataOtherPass(locBoolRegister,locBoolShmem,isMaskFull,isMaskEmpty,blockBeginingX,blockBeginingY,blockBeginingZ,analyzedArr, refAray,resArray,resArraysCounter)
#        CUDA.unsafe_free!(locBool)
        
        
#     end#for
# end#executeDataIter

#numb>>1 & UInt32(1)



"""
uploaded data from shared memory in amask of intrest gets processed in this function so we need to  
    - save it to registers (to locArr)
    - save to the 6 surrounding voxels in shared memory intermediate results 
            - as we also have padding we generally start from spot 2,2 as up and to the left we have 1 padding
            - also we need to make sure that in corner cases we are getting to correct spot
"""
function processMaskData(maskBool::Bool
                         ,zIter::UInt8
                         ,resShmem
                          ) #::CUDA.CuRefValue{Int32}
    # save it to registers - we will need it later
    #locArr[zIter]=maskBool
    #now we are saving results evrywhere we are intrested in so around without diagonals (we use supremum norm instead of euclidean)
    #locArr.x|= maskBool << zIter
    if(maskBool)
        @inbounds resShmem[threadIdxX()+1,threadIdxY()+1,zIter]=true #up
        @inbounds  resShmem[threadIdxX()+1,threadIdxY()+1,zIter+2]=true #down
    
        @inbounds  resShmem[threadIdxX(),threadIdxY()+1,zIter+1]=true #left
        @inbounds   resShmem[threadIdxX()+2,threadIdxY()+1,zIter+1]=true #right

        @inbounds  resShmem[threadIdxX()+1,threadIdxY()+2,zIter+1]=true #front
        @inbounds  resShmem[threadIdxX()+1,threadIdxY(),zIter+1]=true #back
    end#if    
    
end#processMaskData

  


"""
-so we uploaded all data that we consider new - around voxels that are "true"  but we can be sure that some of those were already true earlier 
    possibly it can be marked also by some other neighbouring thread in this particular sweep
    in order to reduce writes to global memory we need to check with registers wheather it is alrerady in a mask - and we will write it to global memory only if it was not
    if the true is in shmem but not in register we write it to global memory - if futher it is also present in other mask (that we are comparing with now)
    we write it also to global result array        
- updata isMaskFull and isMaskEmpty if needed using data from registers and shmem - so later we will know is this mask s full or empty
- we need to take special care for padding - and in case we would find anything there we need to mark appropriate neighbouring block to get activated 
    save result if it did not occured in other mask and write it to global memory

locVal - value from registers
shmemVal - value associated with this thread from shared memory - where we marked neighbours ...
resShmem - shared memory with our preliminary results
isMaskFull, isMaskEmpty - register values needed to specify weather we have full or empty or neither block
blockBeginingX,blockBeginingY,blockBeginingZ - coordinates where our block is begining - will be used as offset by our threads
masktoUpdate - mask that we analyzed and now we write to data about dilatation
maskToCompare - the other mask that we need to check before we write to result array
resArray 
        """
function validataDataFirstPass(locVal::Bool
                    ,shmemVal::Bool
                    ,isMaskFull#::CUDA.CuRefValue{Bool}
                    ,isMaskEmpty#::CUDA.CuRefValue{Bool}
                    ,blockBeginingX::UInt8
                    ,blockBeginingY::UInt8
                    ,blockBeginingZ::UInt8
                    ,maskToCompare
                    ,masktoUpdate
                    ,resArray
                    ,resArraysCounter
                    ,zIter::UInt8)::Bool
    #when this one and previous is true it will still be true


return 

end



function IterToValidate()::Bool
    @unroll for zIteB::UInt8 in UInt8(1):UInt8(32)# most outer loop is responsible for z dimension
        # locBoolRegister::Bool = getLocalBoolRegister(locArr,zIter)
            #  locBoolShmem::Bool = getLocalBoolShemem(resShmem, zIter)
            # ProcessMainData.validataDataFirstPass(locBoolRegister,locBoolShmem,resShmem,isMaskFull,isMaskEmpty,blockBeginingX,blockBeginingY,blockBeginingZ,testArrInn, referenceArray,resArray,resArraysCounter,zIter)
            #CUDA.unsafe_free!(locBoolRegister)
            # CUDA.unsafe_free!(locBoolShmem)
         end#for
     return true    
end

"""
specializes validataDataFirstPass ignoring case of potentially empty mask
iterationNumber - in which iteration we are currently - the bigger it is the higher housedorfrf,,

"""
function validataDataOtherPass(locVal::Bool
                                ,shmemVal::Bool
                                ,isMaskEmpty::MVector{1, Bool}
                                ,blockBeginingX,blockBeginingY,blockBeginingZ
                                ,maskToCompare
                                ,masktoUpdate
                                ,resArray
                                ,iterationNumber::UInt16
                                ,resArraysCounter
                                ,zIter)
    #when this one and previous is true it will still be true
    setIsFull!((locVal | shmemVal),isMaskEmpty  )
    if(!locVal && shmemVal)
    # setting value in global memory
    masktoUpdate[x,y,z+32]= true
    # if we are here we have some voxel that was false in a primary mask and is becoming now true - if it is additionaly true in reference we need to add it to result
    if(maskToCompare[x,y,z+32])
    resArray[x,y,z+32]=iterationNumber
    CUDA.atomic_inc!(pointer(resArraysCounter), UInt16(1))

    end#if
    end#if

end





function setIsFull!(locValOrShmem::Bool
    ,isMaskFull::MVector{1, Bool})
isMaskFull[1]= locValOrShmem & isMaskFull[1]
end#setIsFullOrEmpty


end#ProcessMainData
