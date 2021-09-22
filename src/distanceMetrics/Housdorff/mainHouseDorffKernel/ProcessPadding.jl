"""
process padding in shared memory - where we have results that may affect other blocks
"""
module ProcessPadding
using CUDA, Main.GPUutils, Logging

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
                                    ,locArr)
   #neded to gives shared memory space (we clea and reuse main shared memory) for establishing weather we should process given pading or not 
    #- basically wheather we care about block next to this - so is it full or is it on edge ...
   shmem[threadIdx().x,threadIdx().y,13]= false 
   locArr[1]= false

   isNextBlockOfIntrest(currBlockX,currBlockY,currBlockZ,false,false,false,false,false,true,1,shmem,metadataDims,isPassGold)
   isNextBlockOfIntrest(currBlockX,currBlockY,currBlockZ,false,false,true,false,false,false,3 ,shmem,metadataDims,isPassGold)
   
   isNextBlockOfIntrest(currBlockX,currBlockY,currBlockZ,false,true,false,false,false,false,5,shmem,metadataDims,isPassGold)
   isNextBlockOfIntrest(currBlockX,currBlockY,currBlockZ,false,false,false,false,true,false,7 ,shmem,metadataDims,isPassGold)

   isNextBlockOfIntrest(currBlockX,currBlockY,currBlockZ,false,false,false,true,false,false,9,shmem,metadataDims,isPassGold)
   isNextBlockOfIntrest(currBlockX,currBlockY,currBlockZ,true,false,false,false,false,false,11,shmem,metadataDims,isPassGold)

   sync_threads()

   # so we have 6 planes to analyze all around our data cube and we already collected data wheather are blocks adjacent to those planes are of our intrest
    
   processsPaddingTOP(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z)
   processsPaddingBOTTOM(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z)

   processsPaddingFront(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z)
   processsPaddingBack(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z)

   processsPaddingLeft(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z)
   processsPaddingRight(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z)

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
                            ,isPassGold::Bool)
    #we can ignore all of this if this is edge block - and we can ignore paddings in this situations from those directions, the same for blocks that are full
    processPaddingPlaneAfterCheck(paddingVal,correctedX,correctedY,correctedZ ,sliceNumbManual,shmem
    ,currBlockX+nextBlockXIncrease-nextBlockXDecrease
    ,currBlockY+nextBlockYIncrease-nextBlockYDecrease
    ,currBlockZ+nextBlockZIncrease-nextBlockZDecrease
    ,sourceArray,referenceArray,resArr,metaData,metadataDims,isPassGold)
end#processPaddingPlane


"""
helper function for processPaddingPlane - after passing here test that we care about neighbour
"""
function processPaddingPlaneAfterCheck(paddingVal::Bool
                                        ,correctedX::UInt16
                                        ,correctedY::UInt16
                                        ,correctedZ::UInt16                           
                                        ,sliceNumbManual::UInt8
                                        ,shmem
                                        ,newBlockX::UInt16
                                        ,newBlockY::UInt16
                                        ,newBlockZ::UInt16
                                        ,sourceArray
                                        ,referenceArray
                                        ,resArr
                                        ,metaData
                                        ,metadataDims
                                        ,isPassGold::Bool)

        #before we will start processing padding we need to check weather it makes sense - weather we are not at the edge  or next block is full
        #we need to check is it there  sth of our intrest there - we check shared memory value established in isNextBlockOfIntrestHelper function
        if( shmem[sliceNumbManual,sliceNumbManual,13]  )

           #resetting  main part shared memory for reuse
                shmem[threadIdx().x,threadIdx().y,sliceNumbManual]= false 
                shmem[threadIdx().x,threadIdx().y,sliceNumbManual+1]= false 
                #we are intrested in futher processing only if we have some true here
                if(paddingVal)
                    setGlobalsFromPadding(correctedX,correctedY,correctedZ,resArr,sourceArray)       
                end#if
                #we need to reduce now  the values  of padding vals to establish weather there is any true there if yes we put the neighbour block to be active 
                reducePaddingPlane(shmem,paddingVal,sliceNumbManual )
                #we have true in shmem[1,1,sliceNumbManual+1] if there is at least one true in this plane and false otherwise
                if(shmem[1,1,sliceNumbManual+1])
                    activateNextBlock(newBlockX,newBlockY,newBlockZ, metaData,metadataDims,isPassGold)
                end#if    
  
    end#if 
end #  processPaddingPlaneAfterCheck


"""
    before we will start processing padding we need to check weather it makes sense - weather we are not at the edge  or next block is full
    we will aslo experiment with looking in given direction if all block in this direction are full or empty in both masks - sth to experiment upon
    In order to parallelize the work we will  use separate warps to check the conditions for all  
    we will write results to appropriate spots in shared memory       
"""
function isNextBlockOfIntrest(currBlockX::UInt16
                            ,currBlockY::UInt16
                            ,currBlockZ::UInt16                         
                            ,nextBlockXIncrease::Bool
                            ,nextBlockYIncrease::Bool
                            ,nextBlockZIncrease::Bool
                            ,nextBlockXDecrease::Bool
                            ,nextBlockYDecrease::Bool
                            ,nextBlockZDecrease::Bool
                            ,sliceNumbManual::UInt8
                            ,shmem
                            ,metadataDims
                            ,isPassGold)
                            
                        isNextBlockOfIntrestHelper(currBlockX+nextBlockXIncrease-nextBlockXDecrease
                        ,currBlockY+nextBlockYIncrease-nextBlockYDecrease
                        ,currBlockZ+nextBlockZIncrease-nextBlockZDecrease,sliceNumbManual,shmem,metadataDims,isPassGold)

end#isNextBlockOfIntrest

"""
helper function for isNextBlockOfIntrest
"""
function  isNextBlockOfIntrestHelper(nextBlockX::UInt16
                                    ,nextBlockY::UInt16
                                    ,nextBlockZ::UInt16
                                    ,sliceNumbManual::UInt8
                                    ,shmem
                                    ,metadataDims
                                    ,isPassGold)
  #we are selecting single warp 
  if(threadIdx().y==sliceNumbManual )
    #we need to check is it there at all - it will not if we are in border case
    if(threadIdx().x==1)
        locArr[1]= (nextBlockX>0)
    elseif(threadIdx().x==2)
        locArr[1]= (nextBlockX<=metadataDims[1])
    elseif(threadIdx().x==3)
        locArr[1]= (nextBlockY>0)
    elseif(threadIdx().x==4)
        locArr[1]= (nextBlockY<=metadataDims[2])
    elseif(threadIdx().x==5)
        locArr[1]= (nextBlockZ>0)
    elseif(threadIdx().x==6)
        locArr[1]= (nextBlockZ<=metadataDims[3])
    elseif(threadIdx().x==7)
        locArr[1] = metaData[newBlockX,newBlockY,newBlockZ,isPassGold+3] #then we need to check weather mask is already full - in this case we can not activate it 
    else
        locArr[1] = true # so when all are true we know it 
    end#inner if
    sync_warp()#now we should have all required booleans and we can reduce them

    shmem[sliceNumbManual,sliceNumbManual,13]=reduce_warp_and(locArr[1], UInt8(32))    

end#outer if    

#we will aslo experiment with looking in given direction if all block in this direction are full or empty in both masks we can also 

end#isNextBlockOfIntrest




"""
    activate neighbour  block - invoked when in padding plane we have at least one true
    metaData - basic data about data blocks - stored in 3 dimensional array
    metadataDims - dimensions of metaData array
    isPassGold - true if we are dilatating currently gold standard mask false if we are dilatating other mask

"""
function activateNextBlock(newBlockX,newBlockY,newBlockZ,metadataDims,metaData,isPassGold)
    metaData[newBlockX,newBlockY,newBlockZ,isPassGold+1]= true
end#activateNextBlock    

"""
we need to reduce now  the values  of padding vals to establish weather there is any true there if yes we put the neighbour block to be active 
    if all goes well it should write true to shmem[1,1,sliceNumbManual+1] if there is at leas one true in this plane and false otherwise
"""
function reducePaddingPlane(shmem,paddingVal,sliceNumbManual )::Bool
    #to be experimented is it better to reduce to varyiong y or x 
    @inbounds shmem[1,threadIdx().y,sliceNumbManual]=  reduce_warp_or(paddingVal, UInt8(32))
    #so now we have 32 booleans in shared memory so we need to reduce it one more time using single warp 
    #TODO() check weather warps are column or row wise this below also I am not sure is it good 
    if(threadIdx().y==1)
        @inbounds  shmem[1,1,sliceNumbManual+1]= reduce_warp_or(shmem[1,threadIdx().y,sliceNumbManual], UInt8(32))        
    end    
end




"""
accesses source arrays and modifies it if needed - invoked from procesing of padding
    sourceArrValue - value of source array in x,y,z position
    generally function will be invoked only if value in padding was set to true
"""
function setGlobalsFromPadding(correctedX::UInt16,correctedY::UInt16,correctedZ::UInt16,resArr,sourceArray)
            resArr[correctedX,correctedY,correctedZ]= true
            sourceArray[correctedX,correctedY,correctedZ]= true

end#setGlobalsFromPadding







"""
process Top Padding 
"""
function processsPaddingTOP(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z)
    processPaddingPlane(
        sourceArray[threadIdx().x, threadIdx().y,1],x,y,z-1 # to specialize          
       ,false,false,false,false,false,true # nextBlockXIncrease,nextBlockYIncrease,nextBlockZIncrease,nextBlockXDecrease,nextBlockYDecrease,nextBlockZDecrease
       ,1 # to specialize 
      ,shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold )# this arguments will be the same for all invocations
end


"""
process Bottom Padding 
"""
function processsPaddingBOTTOM(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z)
    processPaddingPlane(
        sourceArray[threadIdx().x, threadIdx().y,34],x,y,z+34          
       ,false,false,true,false,false,false # nextBlockXIncrease,nextBlockYIncrease,nextBlockZIncrease,nextBlockXDecrease,nextBlockYDecrease,nextBlockZDecrease
       ,3 # to specialize 
      ,shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold )# this arguments will be the same for all invocations
end


"""
process  Padding Front 
"""
function processsPaddingFront(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z)
    processPaddingPlane(
        sourceArray[threadIdx().x, 34 ,threadIdx().y],x,y+34,z          
       ,false,true,false,false,false,false # nextBlockXIncrease,nextBlockYIncrease,nextBlockZIncrease,nextBlockXDecrease,nextBlockYDecrease,nextBlockZDecrease
       ,5 # to specialize 
      ,shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold )# this arguments will be the same for all invocations
end

"""
process  Padding back 
"""
function processsPaddingBack(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z)
    processPaddingPlane(
        sourceArray[threadIdx().x, 1 ,threadIdx().y],x,y-1,z          
       ,false,false,false,false,true,false # nextBlockXIncrease,nextBlockYIncrease,nextBlockZIncrease,nextBlockXDecrease,nextBlockYDecrease,nextBlockZDecrease
       ,7 # to specialize 
      ,shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold )# this arguments will be the same for all invocations
end

"""
process  Padding left 
"""
function processsPaddingLeft(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z)
    processPaddingPlane(
        sourceArray[1,threadIdx().x,threadIdx().y],x-1,y,z          
       ,false,false,false,true,false,false # nextBlockXIncrease,nextBlockYIncrease,nextBlockZIncrease,nextBlockXDecrease,nextBlockYDecrease,nextBlockZDecrease
       ,9 # to specialize 
      ,shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold )# this arguments will be the same for all invocations
end

"""
process  Padding right 
"""
function processsPaddingRight(shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold,x,y,z)
    processPaddingPlane(
        sourceArray[34,threadIdx().x,threadIdx().y],x+34,y,z          
       ,true,false,false,false,false,false # nextBlockXIncrease,nextBlockYIncrease,nextBlockZIncrease,nextBlockXDecrease,nextBlockYDecrease,nextBlockZDecrease
       ,11 # to specialize 
      ,shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArr ,metaData,metadataDims,isPassGold )# this arguments will be the same for all invocations
end








end#ProcessPadding