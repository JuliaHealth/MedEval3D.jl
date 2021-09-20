"""
we already converted source array to booleans now we know the number of false positives; false negatives and 
min,max x,y,z of part of data where we have any intreasting maxresultPoints

Now we need to: 
    -populate metadata
        -mark also whether the block is active, foll or neither - from both perspectives of both masks
        -we can set the min,max dims of data block
    -as we already loaded data from global memory - what is supposedly most expensive part - we can do the first dilatation step
    -we prepare rudamentary work plan for  next step    


Toexperiment
    reducePaddingPlane   #to be experimented is it better to reduce to varyiong y or x 



"""
module FirstHouseDorffPass
using CUDA, Main.GPUutils, Logging,StaticArrays

"""
prepare first pass of Housedorff kernel run
datablockDim - the edge length of cube describing the block of data we are working on  - defoult is 32 - which would produce blocks of size 32x32x32
"""
function executefirstHouseDorffPassKernel(datablockDim,metaData)

    threadsNum = (datablockDim,datablockDim)
    blocksNum = size(metaData)



end#executefirstHouseDorffPassKernel
"""


main kernel of muodule it will get 4 data arrays -   reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB
    we work on all simultaneously
    metaData - basic data about data blocks - we will fill it here
    resArrayA, resArrayB- 3 dimensional array of size of reduced source array where entry will be in place where we had covered the voxel, and value will mark in which itwration
        - we have two result arrays separate for each pass    
        - when we will cover some point  we need to report it in the resArray
    localResLastEntryList- results that will be managed by given thread block
    workSchedule - on the basis of it next  phase thread blocks will know on what data blocks to work
    zDim - dimension of a block in z direction - controls the number of iterations we will do by the thread plane
    datablockDim - the edge length  describing the block of data we are working on  - defoult is 32 - which would produce blocks of size 32x32xzDim
        - we will have  the plane of threads of size 32x32 and in order to preserve memory coalescence we should have a warp have the  set of next x positions
    mainArrayDims - dimensions of the main array needed to check for boundary conditions     

"""
function firstHouseDorffPassKernel(  reducedGoldA
                                    ,reducedSegmA
                                    ,reducedGoldB
                                    ,reducedSegmB
                                    ,metaData
                                    ,resArrayA,resArrayB 
                                    ,localResLastEntryList
                                    ,workSchedule
                                    ,datablockDim 
                                    ,mainArrayDims
                                    ,zDim::UInt16 )
    #constant for all threads in a block
    iterationNumber= UInt16(0)

    ############initializations of data 
    resShmem, x,y,z, isMaskFull, isMaskEmpty,locArr = memoryAllocations(datablockDim,zDim)

    
    ############### execution
    @unroll for zIter in UInt16(1):zDim# most outer loop is responsible for z dimension
        processMaskData( reducedGoldA[x,y,z+zIter], zIter, resShmem,locArr )
    end#for 
    sync_threads() #we should have in resShmem what we need 
    @unroll for zIter in UInt16(1):zDim # most outer loop is responsible for z dimension - importnant in this loop we ignore padding we will deal with it separately
        validataData(locArr[zDim],resShmem[threadIdx().x+1,threadIdx().y+1,zIter+1],zDim,resShmem,isMaskFull,isMaskEmpty,x,y,z,reducedGoldA, reducedSegmA,resArrayA,iterationNumber)
    end#for

    #now we need to deal with padding in shmem res



    # now let's check weather block is eligible for futher processing - for this we need sums ...
    #first we check eligibity from gold mask perspective    
     no!! we will modify reduce so it will work on bitwise operators not adding @inbounds shmemSum[wid] = reduce_warp(bitGold,32)

end #FirstHouseDorffPassKernel

"""
this is a little bit tricky to get it correctly as we are looking at paddings from all sides
    - this is also crucial from perspective how we should access indexes of target although we are looking for paddings from all directions
we need to pass data to source array about new "trues"
    paddingVal - value of shared memory associated with this lane
    x,y,z - coordinates of the source arrays - important we need to set those smartly as we are looking at planes all around
    shmem- shared memory boolean array that will be used for reduction as we need to establish weather neighbouring block needs to be activated or not
        - we are reusing main shared memory array 
    nextBlockXchange,nextBlockYchange,nextBlockZchange - indicates wheather we need to add or subtracct from current block ids  in order to get to the  block that is neighbouring from this padding side that we are analyzing
    sourceArray,referenceArray - global memory arrays with primary data  needed to establish weather we need to write data there
    resArr - 3 dimensional array where we put results
    sliceNumbManual - controlls what slice of shared memory we will use       
"""
function processPaddingPlane(paddingVal::Bool
                            ,x::UInt16
                            ,y::UInt16
                            ,z::UInt16
                            ,shmem
                            ,nextBlockXchange::Int8
                            ,nextBlockYchange::Int8
                            ,nextBlockZchange::Int8
                            ,sourceArray
                            ,referenceArray
                            ,resArr
                            ,sliceNumbManual)
    #resetting shared memory for reuse
    shmem[threadIdx().x,threadIdx().y,sliceNumbManual]= false 
    shmem[threadIdx().x,threadIdx().y,sliceNumbManual+1]= false 
    #we are intrested in futher processing only if we have some true here
    if(paddingVal)
        setGlobalsFromPadding(sourceArray,referenceArray,sourceArray[x,y,z] ,x,y,z,resArr)       
    end#if
    #we need to reduce now  the values  of padding vals to establish weather there is any true there if yes we put the neighbour block to be active 
    reducePaddingPlane(shmem,paddingVal,sliceNumbManual )#if all goes well it should write true to shmem[1,1,sliceNumbManual+1] if there is at leas one true in this plane and false otherwise
    if(shmem[1,1,sliceNumbManual+1])
        activateNeighbourBlock(nextBlockXchange,nextBlockYchange, nextBlockZchange)
    end    

end#processPaddingPlane

"""
activate neighbour  block - invoked when in padding plane we have at least one true
"""
function activateNeighbourBlock()
krowa

end#activateNeighbourBlock

"""
we need to reduce now  the values  of padding vals to establish weather there is any true there if yes we put the neighbour block to be active 
    if all goes well it should write true to shmem[1,1,sliceNumbManual+1] if there is at leas one true in this plane and false otherwise
"""
function reducePaddingPlane(shmem,paddingVal,sliceNumbManual )::Bool
    #to be experimented is it better to reduce to varyiong y or x 
    @inbounds shmem[1,threadIdx().y,sliceNumbManual]=  reduce_warp_min(paddingVal, UInt8(32))
    #so now we have 32 booleans in shared memory so we need to reduce it one more time using single warp 
    #TODO() check weather warps are column or row wise this below also I am not sure is it good 
    if(threadIdx().y==1)
        @inbounds  shmem[1,1,sliceNumbManual+1]= reduce_warp_min(shmem[1,threadIdx().y,sliceNumbManual], UInt8(32))        
    end    
end




"""
accesses source arrays and modifies it if needed - invoked from procesing of padding
    sourceArrValue - value of source array in x,y,z position
    generally function will be invoked only if value in padding was set to true
"""
function setGlobalsFromPadding(sourceArray,referenceArray,sourceArrValue::Bool,x::UInt16,y::UInt16,z::UInt16,resArr)
    if(!sourceArrValue)
        sourceArray[x,y,z]=true
        if(referenceArray[x,y,z])
            resArr[x,y,z]= true
        end#if    
end#setGlobalsFromPadding


"""
uploaded data from shared memory in amask of intrest gets processed in this function so we need to  
    - save it to registers (to locArr)
    - save to the 6 surrounding voxels in shared memory intermediate results 
            - as we also have padding we generally start from spot 2,2 as up and to the left we have 1 padding
            - also we need to make sure that in corner cases we are getting to correct spot
"""
function processMaskData(maskBool::Bool
                         ,zIter::UInt16
                         ,resShmem
                         ,locArr )
    # save it to registers - we will need it later
    locArr[zIter]=maskBool
    #now we are saving results evrywhere we are intrested in so around without diagonals (we use supremum norm instead of euclidean)
    if(maskBool)
        resShmem[threadIdx().x+1,threadIdx().y+1,zIter]=true #up
        resShmem[threadIdx().x+1,threadIdx().y+1,zIter+2]=true #down
    
        resShmem[threadIdx().x,threadIdx().y+1,zIter+1]=true #left
        resShmem[threadIdx().x+2,threadIdx().y+1,zIter+1]=true #right

        resShmem[threadIdx().x+1,threadIdx().y+2,zIter+1]=true #front
        resShmem[threadIdx().x+1,threadIdx().y,zIter+1]=true #back
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
zDim - z dimension across which we stream our plane of threads
resShmem - shared memory with our preliminary results
isMaskFull, isMaskEmpty - register values needed to specify weather we have full or empty or neither block
x,y,z - needed to access data from main data array in global memory
masktoUpdate - mask that we analyzed and now we write to data about dilatation
maskToCompare - the other mask that we need to check before we write to result array
iterationNumber - in which iteration we are currently - the bigger it is the higher housedorfrf,,

        """
function validataData(locVal::Bool
                    ,shmemVal::Bool
                     ,zDim::UInt16
                     ,resShmem
                    ,isMaskFull::MVector{1, Bool}
                    ,isMaskEmpty::MVector{1, Bool}
                    ,x::UInt16
                    ,y::UInt16
                    ,z::UInt16
                    ,maskToCompare
                    ,masktoUpdate
                    ,resArray
                    ,iterationNumber::UInt16)
    #when this one and previous is true it will still be true
    setIsFullOrEmpty!((locVal | shmemVal),isMaskFull,isMaskEmpty  )
  if(!locVal && shmemVal)
    # setting value in global memory
    masktoUpdate[x,y,z+zDim]= true
    # if we are here we have some voxel that was false in a primary mask and is becoming now true - if it is additionaly true in reference we need to add it to result
    if(maskToCompare[x,y,z+zDim])
       resArray[x,y,z+zDim]=iterationNumber
    end#if
  end#if

end


"""
set the isMaskFull and isMaskEmpty
locVal - value of the voxel from registry (what was loaded from globl memory and not modified)
shmemVal - what was loaded in shared memory - so after dilatation
yet we pass into locValOrShmem (locVal | shmemVal)
"""
function setIsFullOrEmpty!(locValOrShmem::Bool
                        ,isMaskFull::MVector{1, Bool}
                        ,isMaskEmpty::MVector{1, Bool} )
    isMaskFull[1]= locValOrShmem & isMaskFull[1]
    isMaskEmpty[1] = ~locValOrShmem & isMaskEmpty[1]
end#setIsFullOrEmpty

"""
return basic x,y,z coordinates of thread; z needs to be adjusted for streaming ...
we put it into separate function mainly in order to  cast it to UInt16
"""
function getxyz()::Tuple{UInt16,UInt16,UInt16}
return ((blockIdx().x-1)*32+1,(blockIdx().x-1)*32+1, (blockIdx().z-1)*32+1)
end#getxyz





"""
initialize constants
"""
function memoryAllocations(datablockDim::UInt16,zDim::UInt16)
    #for storing results
    #shmemGold,shmemSegm,shmemSum = createAndInitializeShmem(datablockDim, threadIdx().x)

    krowa - here we have a problem main part will be overwritten but paddings are not 0  - we need to write function to make them 0 !!
    resShmem = CuStaticSharedArray(Bool,(datablockDim+2,datablockDim+2,zDim +2))#+2 in order to get the one padding 
    #coordinates of data in main array
    x,y,z = getxyz()
    #we will use this to establish weather we should mark  the data block as empty or full ...
    isMaskFull= zeros(MVector{1,Bool})
    isMaskEmpty= ones(MVector{1,Bool}) 
    #here we will store in registers data uploaded from mask for later verification wheather we should send it or not
    locArr= zeros(MVector{zDim,Bool})

    return(resShmem, x,y,z, isMaskFull, isMaskEmpty,locArr)

end#initializeMemory

end #FirstHouseDorffPass


# """
# creates shared memory and initializes it to 0
# wid - the number of the warp in the block
# """
# function createAndInitializeShmem(datablockDim, threadIdX)
#    shmemGold = @cuStaticSharedMem(Bool, (datablockDim,datablockDim,datablockDim))
#    shmemSegm = @cuStaticSharedMem(Bool, (datablockDim,datablockDim,datablockDim))
#    #in order to help with summing operations - so we will know wheather block is empty /full ...
#    shmemSums = @cuStaticSharedMem(UInt8,datablockDim)
#    shmemSums[threadIdX] =0
# return (shmemGold,shmemSegm,shmemSums)
# end#createAndInitializeShmem
