"""
process padding in shared memory - where we have results that may affect other blocks
"""
module ProcessPadding
using CUDA, Main.CUDAGpuUtils, Logging

export
"""
Analizing all padding planes - so we need properly pass mainly x,y,z indexes ...

x,y,z - this are coordinates of upper left corner of our data block 
shmem- shared memory boolean array that will be used for reduction as we need to establish weather neighbouring block needs to be activated or not
    - we are reusing main shared memory array 
nextBlockXchange,nextBlockYchange,nextBlockZchange - indicates wheather we need to add or subtracct from current block ids  in order to get to the  block that is neighbouring from this padding side that we are analyzing
currBlockX, Y, Z - points to position of the current block 
sourceArray,referenceArray - global memory arrays with primary data  needed to establish weather we need to write data there
resArr - 3 dimensional array where we put results
metaData - basic data about data blocks - stored in 3 dimensional array
metadataDims - dimensions of metaData array
isPassGold - true if we are dilatating currently gold standard mask false if we are dilatating other mask
resArraysCounter - counter needed to add keep track on how many results we have in our result array
                    - we have separate counter for gold pass and other pass
"""
function processAllPaddingPlanes(    x::UInt16
                                    ,y::UInt16
                                    ,z::UInt16
                                    ,shmem
                                    ,currBlockX::UInt16
                                    ,currBlockY::UInt16
                                    ,currBlockZ::UInt16
                                    ,sourceArray
                                    ,referenceArray
                                    ,resArr
                                    ,metaData
                                    ,metadataDims
                                    ,isPassGold::Bool
                                    ,locArr
                                    ,resArraysCounter
                                    ,iterationNumber)
   #neded to clear memory for establishing weather we should process given pading or not 
    #- basically wheather we care about block next to this - so is it full or is it on edge ...
    locArr = Int32(0)

   isNextBlockOfIntrest(currBlockX,currBlockY,currBlockZ,false,false,false,false,false,true,1,shmem,metadataDims,isPassGold)
   isNextBlockOfIntrest(currBlockX,currBlockY,currBlockZ,false,false,true,false,false,false,3 ,shmem,metadataDims,isPassGold)
   
   isNextBlockOfIntrest(currBlockX,currBlockY,currBlockZ,false,true,false,false,false,false,5,shmem,metadataDims,isPassGold)
   isNextBlockOfIntrest(currBlockX,currBlockY,currBlockZ,false,false,false,false,true,false,7 ,shmem,metadataDims,isPassGold)

   isNextBlockOfIntrest(currBlockX,currBlockY,currBlockZ,false,false,false,true,false,false,9,shmem,metadataDims,isPassGold)
   isNextBlockOfIntrest(currBlockX,currBlockY,currBlockZ,true,false,false,false,false,false,11,shmem,metadataDims,isPassGold)

    for i in [UInt8(1),UInt8(3),UInt8(5),UInt8(7),UInt8(9),UInt8(11)]
        #we need to check weather we are in the same warp as we used in isNextBlockOfIntrest
        if(threadIdxY()==i )
            shmem[i,i,13]=reduce_warp_and(locArr[i], UInt8(32))    
        end#if    
    end#for
    sync_threads()#now we should have all required booleans and we can reduce them
    
   # so we have 6 planes to analyze all around our data cube and we already collected data wheather are blocks adjacent to those planes are of our intrest
    
   processsPaddingTOP(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z,mainQuesCounter,mainWorkQueue,resArraysCounter,iterationNumber)
   processsPaddingBOTTOM(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z,mainQuesCounter,mainWorkQueue,resArraysCounter,iterationNumber)

   processsPaddingFront(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z,mainQuesCounter,mainWorkQueue,resArraysCounter,iterationNumber)
   processsPaddingBack(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z,mainQuesCounter,mainWorkQueue,resArraysCounter,iterationNumber)

   processsPaddingLeft(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z,mainQuesCounter,mainWorkQueue,resArraysCounter,iterationNumber)
   processsPaddingRight(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z,mainQuesCounter,mainWorkQueue,resArraysCounter,iterationNumber)

 end#processAllPaddingPlanes





"""
this is a little bit tricky to get it correctly as we are looking at paddings from all sides
    - this is also crucial from perspective how we should access indexes of target although we are looking for paddings from all directions
we need to pass data to source array about new "trues"
    paddingVal - value of shared memory associated with this lane
    correctedX ,y,z - coordinates of the source arrays - important we need to set those smartly as we are looking at planes all around
    shmem- shared memory boolean array that will be used for reduction as we need to establish weather neighbouring block needs to be activated or not
        - we are reusing main shared memory array 
    nextBlockXchange,nextBlockYchange,nextBlockZchange - indicates wheather we need to add or subtracct from current block ids  in order to get to the  block that is neighbouring from this padding side that we are analyzing
    currBlockX, Y, Z - points to position of the current block 
    sourceArray,referenceArray - global memory arrays with primary data  needed to establish weather we need to write data there
    resArr - 3 dimensional array where we put results
    sliceNumbManual - controlls what slice of shared memory we will use       
    metaData - basic data about data blocks - stored in 3 dimensional array
    metadataDims - dimensions of metaData array
    isPassGold - true if we are dilatating currently gold standard mask false if we are dilatating other mask
"""
function processPaddingPlane(paddingVal::Bool
                            ,isOfIntrest::Bool
                            ,correctedX::UInt16
                            ,correctedY::UInt16
                            ,correctedZ::UInt16                           
                            ,nextBlockXIncrease::Bool
                            ,nextBlockYIncrease::Bool
                            ,nextBlockZIncrease::Bool
                            ,nextBlockXDecrease::Bool
                            ,nextBlockYDecrease::Bool
                            ,nextBlockZDecrease::Bool
                            ,sliceNumbManual::UInt8
                            ,shmem
                            ,currBlockX::UInt16
                            ,currBlockY::UInt16
                            ,currBlockZ::UInt16
                            ,sourceArray
                            ,referenceArray
                            ,resArr
                            ,metaData
                            ,metadataDims
                            ,isPassGold::Bool
                            ,mainQuesCounter
                            ,mainWorkQueue
                            ,resArraysCounter
                            ,iterationNumber)
    #we can ignore all of this if this is edge block - and we can ignore paddings in this situations from those directions, the same for blocks that are full
    newBlockX = currBlockX+nextBlockXIncrease-nextBlockXDecrease
    newBlockY= currBlockY+nextBlockYIncrease-nextBlockYDecrease
    newBlockZ= currBlockZ+nextBlockZIncrease-nextBlockZDecrease

        #before we will start processing padding we need to check weather it makes sense - weather we are not at the edge  or next block is full
        #we need to check is it there  sth of our intrest there - we check shared memory value established in isNextBlockOfIntrestHelper function
        if( shmem[sliceNumbManual,sliceNumbManual,13]  )

            #we are intrested in futher processing only if we have some true here
            if(paddingVal)
                setGlobalsFromPadding(correctedX,correctedY,correctedZ,resArr,sourceArray,resArraysCounter, iterationNumber)       
            end#if
            #we need to reduce now  the values  of padding vals to establish weather there is any true there if yes we put the neighbour block to be active 
            reducePaddingPlane(shmem,paddingVal,sliceNumbManual )
            #we have true in shmem[1,1,sliceNumbManual+1] if there is at least one true in this plane and false otherwise
            if(shmem[1,1,sliceNumbManual+1])
                activateNextBlock(newBlockX,newBlockY,newBlockZ, metaData,metadataDims,isPassGold,mainQuesCounter,mainWorkQueue)
            end#if    

        end#if 
        
end#processPaddingPlane









"""
    activate neighbour  block - invoked when in padding plane we have at least one true
    metaData - basic data about data blocks - stored in 3 dimensional array
    metadataDims - dimensions of metaData array
    isPassGold - true if we are dilatating currently gold standard mask false if we are dilatating other mask
    mainQuesCounter - counter that we will update atomically and will be usefull to populate the work queue
mainWorkQueue - the list of the indicies of  data blocks in metadata with additional information is it referencing the goldpass or second one 


"""
function activateNextBlock(newBlockX,newBlockY,newBlockZ,metadataDims,metaData,isPassGold,mainQuesCounter,mainWorkQueue)
    if(threadIdxY()==1&& threadIdxX()==1)
    metaData[newBlockX,newBlockY,newBlockZ,isPassGold+1]= true
    end
   #so in case it not empty and not full we need to put it into the work queue and increment appropriate counter
    if(threadIdxY()==3&& threadIdxX()==3)
        mainWorkQueue[CUDA.atomic_inc!(pointer(mainQuesCounter), UInt16(1))+1,:]=[newBlockX,newBlockY,newBlockZ,UInt8(isPassGold)] #x,y,z dim of block in metadata
    end#if


end#activateNextBlock    

"""
we need to reduce now  the values  of padding vals to establish weather there is any true there if yes we put the neighbour block to be active 
    if all goes well it should write true to shmem[1,1,sliceNumbManual+1] if there is at leas one true in this plane and false otherwise
"""
function reducePaddingPlane(shmem,paddingVal,sliceNumbManual )::Bool
    #to be experimented is it better to reduce to varyiong y or x 
    @inbounds shmem[1,threadIdxY(),sliceNumbManual]=  reduce_warp_or(paddingVal, UInt8(32))
    #so now we have 32 booleans in shared memory so we need to reduce it one more time using single warp 
    #TODO() check weather warps are column or row wise this below also I am not sure is it good 
    if(threadIdxY()==1  && threadIdxX()==1  )
        @inbounds  shmem[1,1,sliceNumbManual+1]= reduce_warp_or(shmem[1,threadIdxY(),sliceNumbManual], UInt8(32))        
    end    
end




"""
accesses source arrays and modifies it if needed - invoked from procesing of padding
    sourceArrValue - value of source array in x,y,z position
    generally function will be invoked only if value in padding was set to true

resArraysCounter - counter needed to add keep track on how many results we have in our result array
                    - we have separate counter for gold pass and other pass

"""
function setGlobalsFromPadding(correctedX::UInt16,correctedY::UInt16,correctedZ::UInt16,resArr,sourceArray,resArraysCounter, iterationNumber) 
       processResArrBasedOnoldVal(CUDA.atomic_inc!(pointer(resArr[correctedX,correctedY,correctedZ]), UInt16(iterationNumber)),resArraysCounter )
       sourceArray[correctedX,correctedY,correctedZ]= true
end#setGlobalsFromPadding


"""
when we set new result from padding we need to take into account possibility that neighbour block already did it
hence when we set atomic we check the old value if old value was 0 all is good if not we  do not increse  the rescounter
"""
function processResArrBasedOnoldVal(oldVal,resArraysCounter  )
    if(oldVal==0)
        CUDA.atomic_inc!(pointer(resArraysCounter), UInt16(1))
    end    
end



"""
process Top Padding 
"""
function processsPaddingTOP(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z,mainQuesCounter,mainWorkQueue,resArraysCounter,iterationNumber)
    processPaddingPlane(
        sourceArray[threadIdxX(), threadIdxY(),1],x,y,z-1 # to specialize          
       ,false,false,false,false,false,true # nextBlockXIncrease,nextBlockYIncrease,nextBlockZIncrease,nextBlockXDecrease,nextBlockYDecrease,nextBlockZDecrease
       ,1 # to specialize 
      ,shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold ,mainQuesCounter,mainWorkQueue,resArraysCounter,iterationNumber)# this arguments will be the same for all invocations
end


"""
process Bottom Padding 
"""
function processsPaddingBOTTOM(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z,mainQuesCounter,mainWorkQueue,resArraysCounter,iterationNumber)
    processPaddingPlane(
        sourceArray[threadIdxX(), threadIdxY(),34],x,y,z+34          
       ,false,false,true,false,false,false # nextBlockXIncrease,nextBlockYIncrease,nextBlockZIncrease,nextBlockXDecrease,nextBlockYDecrease,nextBlockZDecrease
       ,3 # to specialize 
      ,shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,mainQuesCounter,mainWorkQueue ,resArraysCounter,iterationNumber)# this arguments will be the same for all invocations
end


"""
process  Padding Front 
"""
function processsPaddingFront(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z,mainQuesCounter,mainWorkQueue,resArraysCounter,iterationNumber)
    processPaddingPlane(
        sourceArray[threadIdxX(), 34 ,threadIdxY()],x,y+34,z          
       ,false,true,false,false,false,false # nextBlockXIncrease,nextBlockYIncrease,nextBlockZIncrease,nextBlockXDecrease,nextBlockYDecrease,nextBlockZDecrease
       ,5 # to specialize 
      ,shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,mainQuesCounter,mainWorkQueue ,resArraysCounter,iterationNumber)# this arguments will be the same for all invocations
end

"""
process  Padding back 
"""
function processsPaddingBack(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z,mainQuesCounter,mainWorkQueue,resArraysCounter,iterationNumber)
    processPaddingPlane(
        sourceArray[threadIdxX(), 1 ,threadIdxY()],x,y-1,z          
       ,false,false,false,false,true,false # nextBlockXIncrease,nextBlockYIncrease,nextBlockZIncrease,nextBlockXDecrease,nextBlockYDecrease,nextBlockZDecrease
       ,7 # to specialize 
      ,shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold ,mainQuesCounter,mainWorkQueue,resArraysCounter,iterationNumber)# this arguments will be the same for all invocations
end

"""
process  Padding left 
"""
function processsPaddingLeft(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z,mainQuesCounter,mainWorkQueue,resArraysCounter,iterationNumber)
    processPaddingPlane(
        sourceArray[1,threadIdxX(),threadIdxY()],x-1,y,z          
       ,false,false,false,true,false,false # nextBlockXIncrease,nextBlockYIncrease,nextBlockZIncrease,nextBlockXDecrease,nextBlockYDecrease,nextBlockZDecrease
       ,9 # to specialize 
      ,shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,mainQuesCounter,mainWorkQueue ,resArraysCounter,iterationNumber)# this arguments will be the same for all invocations
end

"""
process  Padding right 
"""
function processsPaddingRight(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z,mainQuesCounter,mainWorkQueue,resArraysCounter,iterationNumber)
    processPaddingPlane(
        sourceArray[34,threadIdxX(),threadIdxY()],x+34,y,z          
       ,true,false,false,false,false,false # nextBlockXIncrease,nextBlockYIncrease,nextBlockZIncrease,nextBlockXDecrease,nextBlockYDecrease,nextBlockZDecrease
       ,11 # to specialize 
      ,shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,mainQuesCounter,mainWorkQueue,resArraysCounter,iterationNumber )# this arguments will be the same for all invocations
end








end#ProcessPadding



# """
#     before we will start processing padding we need to check weather it makes sense - weather we are not at the edge  or next block is full
#     we will aslo experiment with looking in given direction if all block in this direction are full or empty in both masks - sth to experiment upon
#     In order to parallelize the work we will  use separate warps to check the conditions for all  
#     we will write results to appropriate spots in shared memory       
# """
# function isNextBlockOfIntrest(currBlockX::UInt16
#                             ,currBlockY::UInt16
#                             ,currBlockZ::UInt16                         
#                             ,nextBlockXIncrease::Bool
#                             ,nextBlockYIncrease::Bool
#                             ,nextBlockZIncrease::Bool
#                             ,nextBlockXDecrease::Bool
#                             ,nextBlockYDecrease::Bool
#                             ,nextBlockZDecrease::Bool
#                             ,sliceNumbManual::UInt8
#                             ,shmem
#                             ,metadataDims
#                             ,isPassGold)
#        nextBlockX = currBlockX+nextBlockXIncrease-nextBlockXDecrease
#        nextBlockY = currBlockY+nextBlockYIncrease-nextBlockYDecrease
#        nextBlockZ = currBlockZ+nextBlockZIncrease-nextBlockZDecrease                     

# #we are selecting single warp 
# if(threadIdxY()==sliceNumbManual )
#     #we need to check is it there at all - it will not if we are in border case
#     if(threadIdxX()==1)
#         locArr[sliceNumbManual]= (nextBlockX>0)
#     elseif(threadIdxX()==2)
#         locArr[sliceNumbManual]= (nextBlockX<=metadataDims[1])
#     elseif(threadIdxX()==3)
#         locArr[sliceNumbManual]= (nextBlockY>0)
#     elseif(threadIdxX()==4)
#         locArr[sliceNumbManual]= (nextBlockY<=metadataDims[2])
#     elseif(threadIdxX()==5)
#         locArr[sliceNumbManual]= (nextBlockZ>0)
#     elseif(threadIdxX()==6)
#         locArr[sliceNumbManual]= (nextBlockZ<=metadataDims[3])
#     elseif(threadIdxX()==7)
#         locArr[sliceNumbManual] = metaData[newBlockX,newBlockY,newBlockZ,isPassGold+3] #then we need to check weather mask is already full - in this case we can not activate it 
#     else
#         locArr[sliceNumbManual] = true # so when all are true we know it 
#     end#inner if

# end#outer if    

# #we will aslo experiment with looking in given direction if all block in this direction are full or empty in both masks we can also 
