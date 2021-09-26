

"""
loads and do the main processing of data in arrays of intrest (padding of shmem will be analyzed separately)

"""
module ProcessMainData
using  StaticArrays,Main.CUDAGpuUtils ,Main.HFUtils, CUDA
export executeDataIterFirstPass,executeDataIterOtherPasses,processMaskData


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
function executeDataIterFirstPass(analyzedArr, refAray,blockBeginingX,blockBeginingY,blockBeginingZ,isMaskFull,isMaskEmpty,resShmem,locArr,resArray)
    @unroll for zIter in UInt16(1):32# most outer loop is responsible for z dimension
        local locBool = testArrInn[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]
        locArr|= locBool << zIter
       processMaskData( locBool, zIter, resShmem)
       CUDA.unsafe_free!(locBool)
    end#for 
    sync_threads() #we should have in resShmem what we need 
    @unroll for zIter in UInt16(1):32 # most outer loop is responsible for z dimension - importnant in this loop we ignore padding we will deal with it separately
          local locBoolRegister = (locArr>>zIter & UInt32(1))==UInt32(1)
          local locBoolShmem = resShmem[threadIdxX()+1,threadIdxY()+1,zIter+1]
        validataDataFirstPass(locBoolRegister,locBoolShmem,resShmem,isMaskFull,isMaskEmpty,blockBeginingX,blockBeginingY,blockBeginingZ,analyzedArr, refAray,resArray,resArraysCounter,zIter)
    end#for
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
    locValOrShmem = (locVal | shmemVal)
    isMaskFull.x= locValOrShmem & isMaskFull.x
    isMaskEmpty.x = ~locValOrShmem & isMaskEmpty.x

  if(!locVal && shmemVal)
    # setting value in global memory
    @inbounds  masktoUpdate[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]= true
    # if we are here we have some voxel that was false in a primary mask and is becoming now true - if it is additionaly true in reference we need to add it to result
    if(maskToCompare[@inbounds  (blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)])
        @inbounds  resArray[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]=UInt16(1)
       
       atomicallyAddOneUint32(resArraysCounter)
    end#if
  end#if
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