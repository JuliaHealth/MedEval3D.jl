

"""
loads and do the main processing of data in arrays of intrest (padding of shmem will be analyzed separately)

"""
module ProcessMainData

export executeDataIter


"""
loads and do the main processing of data in arrays of intrest (padding of shmem will be analyzed separately)
zDim - how far we want to iterate in z dimension - ussually 32
analyzedArr - array we are currently dilatating
refAray -array we are referencing (we do not dilatate it only check against it )
iterationNumber - at what iteration of dilatation we are - so how many dilatations we already performed
x,y,z - the x,y,z coordinates of the begining of this block
"""
function executeDataIter(zDim,analyzedArr, refAray,iterationNumber ,x::UInt16,y::UInt16,z::UInt16)
    @unroll for zIter in UInt16(1):zDim# most outer loop is responsible for z dimension
        processMaskData( analyzedArr[x,y,z+zIter], zIter, resShmem,locArr,isMaskFull,isMaskEmpty,resShmem )
    end#for 
    sync_threads() #we should have in resShmem what we need 
    @unroll for zIter in UInt16(1):zDim # most outer loop is responsible for z dimension - importnant in this loop we ignore padding we will deal with it separately
        validataData(locArr[zDim],resShmem[threadIdx().x+1,threadIdx().y+1,zIter+1],zDim,resShmem,isMaskFull,isMaskEmpty,x,y,z,analyzedArr, refAray,resArrayA,iterationNumber)
    end#for
end#executeDataIter



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

end#ProcessMainData