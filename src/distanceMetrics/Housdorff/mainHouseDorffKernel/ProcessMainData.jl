

"""
loads and do the main processing of data in arrays of intrest (padding of shmem will be analyzed separately)

"""
module ProcessMainData
using  StaticArrays,Main.CUDAGpuUtils ,Main.HFUtils, CUDA, Main.ProcessPadding
export executeDataIterFirstPass,executeDataIterOtherPasses,processMaskData,executeDataIterFirstPassWithPadding


"""
loads and do the main processing of data in arrays of intrest (padding of shmem will be analyzed separately)
analyzedArr - array we are currently dilatating
refAray -array we are referencing (we do not dilatate it only check against it )
iterationNumber - at what iteration of dilatation we are - so how many dilatations we already performed
blockBeginingX,blockBeginingY,blockBeginingZ - coordinates where our block is begining - will be used as offset by our threads
isMaskFull,isMaskEmpty - enables later checking is mask is empty full or neither
resShmem - shared memory 34x34x34 bit array
locArr - local bit array of thread
resArray- 3 dimensional array where we put results
"""
function executeDataIterFirstPass(analyzedArr, referenceArray,blockBeginingX,blockBeginingY,blockBeginingZ,resShmem,resArray,resArraysCounter)
    locArr = Int32(0)
    isMaskFull= true
    isMaskEmpty = true
    #locArr.x |= true << UInt8(2)

    ############## upload data
    @unroll for zIter::UInt8 in UInt8(1):UInt8(32)# most outer loop is responsible for z dimension
        locBool::Bool = @inbounds analyzedArr[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]
        locArr|= locBool << zIter
        # CUDA.@cuprint "locBool $(locBool)  locArr $(locArr) \n  "
        processMaskData( locBool, zIter, resShmem)
    end#for 
   sync_threads() #we should have in resShmem what we need 
    ########## check data aprat from padding
    @unroll for zIter::UInt8 in UInt8(1):UInt8(32)# most outer loop is responsible for z dimension
        locVal::Bool = @inbounds  (locArr>>zIter & Int32(1))==Int32(1)
        shmemVal::Bool = @inbounds resShmem[threadIdxX()+1,threadIdxY()+1,zIter+1]
        # CUDA.@cuprint "locVal $(locVal)  shmemVal $(shmemVal) \n  "
        locValOrShmem = (locVal | shmemVal)
        isMaskFull= locValOrShmem & isMaskFull
        isMaskEmpty = ~locValOrShmem & isMaskEmpty

        #CUDA.@cuprint "locVal $(locVal)  shmemVal $(shmemVal) \n  "
        if(!locVal && shmemVal)
            # setting value in global memory
            @inbounds  analyzedArr[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]= true
            # if we are here we have some voxel that was false in a primary mask and is becoming now true - if it is additionaly true in reference we need to add it to result
            isInReferencaArr::Bool= @inbounds referenceArray[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]
            if(isInReferencaArr)
                #CUDA.@cuprint "isInReferencaArr $(isInReferencaArr) \n  "
                @inbounds  resArray[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]=UInt16(1)
                #CUDA.atomic_inc!(pointer(resArraysCounter), Int32(1))              
                atomicallyAddOneInt(resArraysCounter)
            end#if
        end#if
      end#for

end#executeDataIter















function executeDataIterFirstPassWithPadding(analyzedArr, referenceArray,blockBeginingX
                                ,blockBeginingY,blockBeginingZ,resShmem,resArray,resArraysCounter
                                ,currBlockX,currBlockY,currBlockZ,isPassGold,metaData,metadataDims
                                ,mainQuesCounter,mainWorkQueue,iterationNumber)
    locArr = Int32(0)
    isMaskFull= true
    isMaskEmpty = true
    isMaskOkForProcessing = true
    offset = UInt16(1)
    nextBlockX,nextBlockY, nextBlockZ = UInt8(0), UInt8(0),UInt8(0)
    #locArr.x |= true << UInt8(2)

    ############## upload data
    @unroll for zIter::UInt8 in UInt8(1):UInt8(32)# most outer loop is responsible for z dimension
        locBool::Bool = @inbounds analyzedArr[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]
        locArr|= locBool << zIter
        # CUDA.@cuprint "locBool $(locBool)  locArr $(locArr) \n  "
        processMaskData( locBool, zIter, resShmem)
    end#for 
   sync_threads() #we should have in resShmem what we need 
    ########## check data aprat from padding
    @unroll for zIter::UInt8 in UInt8(1):UInt8(32)# most outer loop is responsible for z dimension
        locVal::Bool = @inbounds  (locArr>>zIter & Int32(1))==Int32(1)
        shmemVal::Bool = @inbounds resShmem[threadIdxX()+1,threadIdxY()+1,zIter+1]
        locValOrShmem = (locVal | shmemVal)
        isMaskFull= locValOrShmem & isMaskFull
        isMaskEmpty = ~locValOrShmem & isMaskEmpty

        if(!locVal && shmemVal)
            # setting value in global memory
            @inbounds  analyzedArr[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]= true
            # if we are here we have some voxel that was false in a primary mask and is becoming now true - if it is additionaly true in reference we need to add it to result
            isInReferencaArr::Bool= @inbounds referenceArray[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]
            if(isInReferencaArr)
                @inbounds  resArray[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]=UInt16(1)
                atomicallyAddOneInt(resArraysCounter)
            end#if
        end#if
      end#for
     ################################################################################################################################ 
     #processing padding
     function isNextBlockOfIntrest()



        HFUtils.clearMainShmem(resShmem)

        # first we check weather next block is viable for processing
        @unroll for zIter::UInt8 in UInt8(1):UInt8(6)
            #we will iterate over all padding planes below way to calculate the next block in all dimensions not counting oblique directions
            nextBlockX = currBlockX+(zIter==1)-(zIter==2)
            nextBlockY = currBlockY+(zIter==3)-(zIter==4)
            nextBlockZ = currBlockZ+(zIter==5)-(zIter==6)

            @ifXY 1 zIter @inbounds isMaskOkForProcessing = nextBlockX>0
            @ifXY 2 zIter @inbounds isMaskOkForProcessing = nextBlockX<=metadataDims[1]
            @ifXY 3 zIter @inbounds isMaskOkForProcessing = nextBlockY>0
            @ifXY 4 zIter @inbounds isMaskOkForProcessing = nextBlockY<=metadataDims[2]
            @ifXY 5 zIter @inbounds isMaskOkForProcessing = nextBlockZ>0
            @ifXY 6 zIter @inbounds isMaskOkForProcessing = nextBlockZ<=metadataDims[3]
            @ifXY 7 zIter @inbounds isMaskOkForProcessing = metaData[nextBlockX,nextBlockZ,newBlockZ,isPassGold+3]#then we need to check weather mask is already full - in this case we can not activate it 
            #now we check are all true 
            @ifY zIter begin 
                    while(offset <UInt16(8)) 
                    isMaskOkForProcessing = isMaskOkForProcessing & shfl_down_sync(FULL_MASK, isMaskOkForProcessing, offset)
                    offset<<= 1
                end #while
            end# @ifY 
            #here is the information wheather we want to process next block
            @ifXY zIter 1 @inbounds resShmem[2,zIter+1,2]
        end#for zIter   
        sync_threads()#now we should know wheather we are intrested in blocks around
        isMaskOkForProcessing=false#reset for reuse
        
        # ################################################################################################################################ 
        #checking is there anything in the padding plane - so we basically do (most of reductions)
        if(resShmem[2,zIter+1,2] )
            @unroll for zIter::UInt8 in UInt8(1):UInt8(6)       
                #we need to reduce now  the values  of padding vals to establish weather there is any true there if yes we put the neighbour block to be active 
                    #reduction
                    offset = UInt16(1)
                    while(offset <32) 
                        isMaskOkForProcessing = isMaskOkForProcessing | shfl_down_sync(FULL_MASK, isMaskOkForProcessing, offset)  
                        offset<<= 1
                    end
                    @ifX 1  @inbounds  resShmem[zIter+1,threadIdxY()+1,3]=isMaskOkForProcessing
            end
             sync_threads()#now we have partially reduced values marking wheather we have any true in padding
            # we get full reductions
            @unroll for zIter::UInt8 in UInt8(1):UInt8(6)
                offset = UInt16(1)
                if(threadIdxY()==zIter)
                    while(offset <32)                        
                        @inbounds  resShmem[zIter+1,threadIdxX()+1,3] = shfl_down_sync(FULL_MASK,resShmem[zIter+1,threadIdxX()+1,3], offset)
                    end#while    
                end#if    
            end#for
         end#if  
         sync_threads()#now we have fully reduced in resShmem[zIter+1,1+1,3]= resShmem[zIter+1,2,3]

         CUDA.@cuprint "resShmem[zIter+1,threadIdxX()+1,3]" 




    end#isNextBlockOfIntrest
    
   # isNextBlockOfIntrest() krowa

     ################################################################################################################################ 


    #function will be invoked on paddings from all sides
    function innerProcessPadding(   resShmemX::UInt8 # x value to find padding value of intrest in shared memory
                                    ,resShmemY::UInt8 # y value to find padding value of intrest in shared memory
                                    ,resShmemZ::UInt8 # z value to find padding value of intrest in shared memory
                                    ,primaryZiter::UInt8 # the same as we analyzed above what blocks should be analyzed
                                    ,correctedX::UInt16 # x value to find padding value of intrest in global memory
                                    ,correctedY::UInt16# y value to find padding value of intrest in global memory
                                    ,correctedZ::UInt16)# z value to find padding value of intrest in global memory
        sync_warp()# to reduce warp divergence

        nextBlockX = currBlockX+(zIter==1)-(zIter==2)
        nextBlockY = currBlockY+(zIter==3)-(zIter==4)
        nextBlockZ = currBlockZ+(zIter==5)-(zIter==6)
        #getting value from padding
        @inbounds paddingVal= resShmem[resShmemX,resShmemY,resShmemZ ]
        #whether we should analyze next block
        if(resShmem[2,primaryZiter+1,2] && resShmem[primaryZiter+1,2,3] )
            #activating next block
            @ifXY 1 primaryZiter metaData[nextBlockX,nextBlockY,nextBlockZ,UInt8(isPassGold+1)]= true
            #increasing mainWorkQueue counter and using its old value we know where we should update the mainWorkQueue with newly activated block
            @ifXY 2 primaryZiter mainWorkQueue[atomicallyAddOneInt(mainQuesCounter)+1,:]=[newBlockX,newBlockY,newBlockZ,UInt8(isPassGold)] #x,y,z dim of block in metadata
            # when we set new result from padding we need to take into account possibility that neighbour block already did it
            # hence when we set atomic we check the old value if old value was 0 all is good if not we  do not increse  the rescounter
            @ifXY 3 primaryZiter begin
                old = atomicallySetValueTrreeDim(resArr,correctedX,correctedY,correctedZ,iterationNumber)
                if(old==0)
                    atomicallyAddOneInt(resArrayCounter)
                end    
            end   
        end#if
    end#innerProcessPadding
    
    ######################################## 
    #now we will analyze the padding planes one by one using innerProcessPadding

    ######## TOP
#     innerProcessPadding(threadIdxX()#resShmemX
#                         ,threadIdxY()#resShmemY
#                         ,1#resShmemZ
#                         ,5 #primaryZiter
#                         ,blockBeginingX+threadIdxX()#correctedX
#                         ,blockBeginingY +threadIdxY()#correctedY
#                         ,blockBeginingZ-1#correctedZ
#                         )
   
#    ######## BOTTOM
#    innerProcessPadding(threadIdxX()#resShmemX
#                         ,threadIdxY()#resShmemY
#                         ,34#resShmemZ
#                         ,6 #primaryZiter
#                         ,blockBeginingX+threadIdxX()#correctedX
#                         ,blockBeginingY +threadIdxY()#correctedY
#                         ,blockBeginingZ+33#correctedZ
#                         )
#    ######## LEFT
#    innerProcessPadding(1#resShmemX
#                         ,threadIdxY()#resShmemY
#                         ,threadIdxX()#resShmemZ
#                         ,2 #primaryZiter
#                         ,blockBeginingX-1#correctedX
#                         ,blockBeginingY +threadIdxX()#correctedY
#                         ,blockBeginingZ+threadIdxY()#correctedZ
#                         )
#    ######## RIGHT
#    innerProcessPadding(1#resShmemX
#                         ,threadIdxY()#resShmemY
#                         ,threadIdxX()#resShmemZ
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

function myIncreaseBitt(maskBool::Bool,zIter::UInt8, locArr::CUDA.CuRefValue{Int32} )::Bool
    locArr.x|= maskBool << zIter
    return true
end    


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