
"""
after all processing we need to establish weather the mask is full  (in case of first iteration also is it empty)
"""
module IsBlockToStayActive
using CUDA, Main.GPUutils, Logging,StaticArrays,Main.HFUtils
export isActiveForFirstPass, isActiveForNormalPass
"""
We check is block still active in first pass - we need to check is empty or full and set the metadata
    for reference below te fourth dimension of metadata 
        1) isActiveOrFullForPasssegm - true if other  mask is acive for modifications  
        2) isActiveOrFullForPassgold -  true if gold standard mask is acive for modifications 
        3) isFullSegm - true if other mask is full (only ones)
        4) isFullGold - true if gold standard mask is full (only ones)
"""
function isActiveForFirstPass(isMaskFull::MVector{1,Bool}
                            , isMaskEmpty::MVector{1,Bool}
                            ,shmem
                            ,currBlockX::UInt16
                            ,currBlockY::UInt16
                            ,CurrBlockZ::UInt16
                            ,isPassGold::Bool
                            ,metaData )
    @inbounds shmem[1,threadIdx().y,20]=  reduce_warp_and(isMaskFull, UInt8(32))
    @inbounds shmem[1,threadIdx().y,21]=  reduce_warp_and(isMaskEmpty, UInt8(32))
    #so now we have 32 booleans in shared memory so we need to reduce it one more time using single warp 
    #TODO() check weather warps are column or row wise this below also I am not sure is it good  

    if(threadIdx().y==1 && threadIdx().x==1)
        @inbounds  shmem[1,1,22]= reduce_warp_and(shmem[1,threadIdx().y,20], UInt8(32))  #true if mask is full    
    end  
    if(threadIdx().y==2 && threadIdx().x==2)
        @inbounds  shmem[1,2,22]= reduce_warp_and(shmem[1,threadIdx().y,21], UInt8(32))  #true if mask is empty      
    end 
    sync_threads()
    #we have needed data in shmem
    if(threadIdx().y==1 && threadIdx().x==1 && (shmem[1,2,22] || shmem[1,1,22]))
        metaData[currBlockX,currBlockY,CurrBlockZ,isPassGold+1]=false # we set is inactive 
        metaData[currBlockX,currBlockY,CurrBlockZ,isPassGold+3]=true # we set is as full

    end#if
end#isActiveForFirstPass   



"""
We check is block still active in first pass - we need to check is full (if it would be empty we would not start to analyze it after first pass)
"""
function isActiveForNormalPass(isMaskFull)
    @inbounds shmem[1,threadIdx().y,20]=  reduce_warp_and(isMaskFull, UInt8(32))
    #so now we have 32 booleans in shared memory so we need to reduce it one more time using single warp 
    #TODO() check weather warps are column or row wise this below also I am not sure is it good 
    if(threadIdx().y==1 && threadIdx().x==1)
        @inbounds  shmem[1,1,22]= reduce_warp_and(shmem[1,threadIdx().y,20], UInt8(32))  #true if mask is full    
    end  
    sync_threads()
    #we have needed data in shmem
    if(threadIdx().y==1 && threadIdx().x==1 && shmem[1,2,22])
        metaData[currBlockX,currBlockY,CurrBlockZ,isPassGold+1]=false # we set is inactive 
        metaData[currBlockX,currBlockY,CurrBlockZ,isPassGold+3]=true # we set is as full
    end#if
end#isActiveForNormalPass

end#IsBlockToStayActive