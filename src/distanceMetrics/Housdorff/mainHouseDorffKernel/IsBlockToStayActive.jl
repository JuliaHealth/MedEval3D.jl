
"""
after all processing we need to establish weather the mask is full  (in case of first iteration also is it empty)
"""
module IsBlockToStayActive
using CUDA, ..CUDAGpuUtils, Logging,StaticArrays,..HFUtils
export isActiveForFirstPass, isActiveForNormalPass
"""
We check is block still active in first pass - we need to check is empty or full and set the metadata
    for reference below te fourth dimension of metadata 
        1) isActiveOrFullForPasssegm - true if other  mask is acive for modifications  
        2) isActiveOrFullForPassgold -  true if gold standard mask is acive for modifications 
        3) isFullSegm - true if other mask is full (only ones)
        4) isFullGold - true if gold standard mask is full (only ones)

        shmem- shared memory
        metaData - 3 dim array with metadata of data blocks
        currMatadataBlockX,currMatadataBlockY, currMatadataBlockZ - cartesian coordinates of current block in metadaa!!!
        mainQuesCounter - counter that we will update atomically and will be usefull to populate the work queue
        mainWorkQueue - the list of the indicies of  data blocks in metadata with additional information is it referencing the goldpass or second one 

"""
function isActiveForFirstPass(isMaskFull::MVector{1,Bool}
                            , isMaskEmpty::MVector{1,Bool}
                            ,shmem
                            ,currMatadataBlockX::UInt8
                            ,currMatadataBlockY::UInt8
                            ,currMatadataBlockZ::UInt8
                            ,isPassGold::Bool
                            ,metaData
                            ,mainQuesCounter
                            ,mainWorkQueue )
    @inbounds shmem[1,threadIdxY(),20]=  reduce_warp_and(isMaskFull, UInt8(32))
    @inbounds shmem[1,threadIdxY(),21]=  reduce_warp_and(isMaskEmpty, UInt8(32))
    #so now we have 32 booleans in shared memory so we need to reduce it one more time using single warp 
    #TODO() check weather warps are column or row wise this below also I am not sure is it good  

    if(threadIdxY()==1 && threadIdxX()==1)
        @inbounds  shmem[1,1,22]= reduce_warp_and(shmem[1,threadIdxY(),20], UInt8(32))  #true if mask is full    
    end  
    if(threadIdxY()==2 && threadIdxX()==2)
        @inbounds  shmem[1,2,22]= reduce_warp_and(shmem[1,threadIdxY(),21], UInt8(32))  #true if mask is empty      
    end 
    sync_threads()
    #we have needed data in shmem

    # we update matedata concurrently
    if(threadIdxY()==1 && threadIdxX()==1 && (shmem[1,2,22] || shmem[1,1,22]))
        metaData[currMatadataBlockX,currMatadataBlockY,currMatadataBlockZ,isPassGold+1]=false # we set is inactive 
    end#if   
    if(threadIdxY()==1 && threadIdxX()==1 && (shmem[1,2,22] || shmem[1,1,22]))
        metaData[currMatadataBlockX,currMatadataBlockY,currMatadataBlockZ,isPassGold+3]=true # we set is as full
    end#if
    #so in case it not empty and not full we need to put it into the work queue and increment appropriate counter
    if(threadIdxY()==3&& threadIdxX()==3 && !(shmem[1,2,22] || shmem[1,1,22]))
        mainWorkQueue[CUDA.atomic_inc!(pointer(mainQuesCounter), UInt16(1))+1]=[currMatadataBlockX,currMatadataBlockY,currMatadataBlockZ,UInt8(isPassGold)] #x dim of block in metadata
    end#if

end#isActiveForFirstPass   



"""
We check is block still active in first pass - we need to check is full (if it would be empty we would not start to analyze it after first pass)
shmem- shared memory
metaData - 3 dim array with metadata of data blocks
currMatadataBlockX,currMatadataBlockY, currMatadataBlockZ - cartesian coordinates of current block in metadaa!!!
mainQuesCounter - counter that we will update atomically and will be usefull to populate the work queue
mainWorkQueue - the list of the indicies of  data blocks in metadata with additional information is it referencing the goldpass or second one 
workingQueuePosition - points out to the index of currently analyzed spot in work queue that we are in order to 

"""
function isActiveForNormalPass(isMaskFull::MVector{1,Bool}
                                ,shmem
                                ,currMatadataBlockX::UInt8
                                ,currMatadataBlockY::UInt8
                                ,currMatadataBlockZ::UInt8
                                ,isPassGold::Bool
                                ,metaData
                                ,mainQuesCounter
                                ,mainWorkQueue )
    @inbounds shmem[1,threadIdxY(),20]=  reduce_warp_and(isMaskFull, UInt8(32))
    #so now we have 32 booleans in shared memory so we need to reduce it one more time using single warp 
    #TODO() check weather warps are column or row wise this below also I am not sure is it good 
    if(threadIdxY()==1 && threadIdxX()==1)
        @inbounds  shmem[1,1,22]= reduce_warp_and(shmem[1,threadIdxY(),20], UInt8(32))  #true if mask is full    
    end  
    sync_threads()
    #we have needed data in shmem
    if(threadIdxY()==1 && threadIdxX()==1 && (shmem[1,2,22]))
        metaData[currMatadataBlockX,currMatadataBlockY,currMatadataBlockZ,isPassGold+1]=false # we set is inactive 
    end#if   
    if(threadIdxY()==2 && threadIdxX()==1 && (shmem[1,2,22]))
        metaData[currMatadataBlockX,currMatadataBlockY,currMatadataBlockZ,isPassGold+3]=true # we set is as full
    end#if

    #so in case it not empty and not full we need to put it into the work queue and increment appropriate counter
    if(threadIdxY()==3&& threadIdxX()==3 && !(shmem[1,2,22]))
        mainWorkQueue[CUDA.atomic_inc!(pointer(mainQuesCounter), UInt16(1))+1]=[currMatadataBlockX,currMatadataBlockY,currMatadataBlockZ,UInt8(isPassGold)] #x dim of block in metadata
    end#if

end#isActiveForNormalPass

end#IsBlockToStayActive