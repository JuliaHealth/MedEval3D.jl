module WorkQueueUtils
using  ..BitWiseUtils,CUDA, Logging,..CUDAGpuUtils, ..ResultListUtils,..WorkQueueUtils, Logging,StaticArrays, ..IterationUtils, ..ReductionUtils, ..CUDAAtomicUtils,..MetaDataUtils, ..BitWiseUtils
export allocateWorkQueue,appendToWorkQueue,@appendToWorkQueueBasic,@appendToLocalWorkQueue,@appendToGlobalWorkQueue
"""
allocate memory for  work queues
    entries means
        1)xMeta
        2)yMeta
        3)zMeta
        4)isGold
 In order to prevent overwriting  we will create 8 separe work queues for each even or odd metax,metay and meta Z ... 
    also initializes counters for each 
    names are encoded like E is about even and O for odd so workQueueEEE will mean that all metaX,metaY and metaZ are even    
   returns  workQueue,workQueueCounter
    
    """
function allocateWorkQueue(metaDataLength)
    queueSize = cld(metaDataLength*2,8)+2#*2 becouse of gold and segm pass divided by 8 becouse we have 8 work queues
    return (  CUDA.zeros(UInt16,4,(queueSize)),CUDA.zeros(UInt16,(1)) )
    

    
end

"""
atomically append the block linear index and information is it gold or other pass 
also we need to be sure that we appended to the correct work queue based on the properties of the xMeta,yMeta,zMeta - so are they even, odd ...
"""
macro appendToLocalWorkQueue(metaX,metaY,metaZ,isGold ) 
    return esc(quote 
    old =  atomicallyAddOne(workCounterLocalInShmem)
      #CUDA.@cuprint " ooo old $(old) metaX $($metaX) metaY $($metaY) metaZ $($metaZ) isGold $($isGold) workCounterLocalInShmem[1] $(workCounterLocalInShmem[1]) \n"
    @inbounds shmemblockData[old*4+1]= $metaX
    @inbounds shmemblockData[old*4+2]= $metaY
    @inbounds shmemblockData[old*4+3]= $metaZ
    @inbounds shmemblockData[old*4+4]= $isGold

    # workQueue[1,old]= $metaX
    # workQueue[2,old]= UInt16($metaY)
    # workQueue[3,old]= UInt16($metaZ)
    # workQueue[4,old]= UInt16($isGold)


    end)#qote 
end#appendToWorkQueue

"""
the function above is block private now we need to load the work queaue into main work queue
    and additionally atomically add to general work queue counter
"""
macro appendToGlobalWorkQueue() 
    return esc(quote 
    if(workCounterLocalInShmem[1]>0)
        #old is a spot where we should start adding new entries
        sync_threads()
        @ifXY 1 1 workCounterHelper[1] =  atomicallyAddToSpot(workQueueCounter,1,workCounterLocalInShmem[1])-workCounterLocalInShmem[1]
        sync_threads()
        # @ifXY 1 1 CUDA.@cuprint "appendToGlobalWorkQueue old $(old)  workCounterLocalInShmem[1] $(workCounterLocalInShmem[1]) \n "
        #we multiply it by 4 so each thread will have potentially only one data entry to do 
        @iterateLinearlyCheckAll(cld(workCounterLocalInShmem[1]*4,blockDimX()*blockDimY()),workCounterLocalInShmem[1]*4,begin   
            @inbounds workQueue[(workCounterHelper[1])*4+i] =  shmemblockData[i]
        end)
    end

end)#qote 
end#appendToWorkQueue



# macro appendToWorkQueueBasic(workQueaue,workQueauecounter, metaX,metaY,metaZ,isGold ) 
#     return esc(quote 
#     old =  atomicallyAddOne($workQueauecounter)+1
#     # CUDA.@cuprint " ooo old $(old) metaX $($metaX) metaY $($metaY) metaZ $($metaZ) isGold $($isGold) workQueaue $(length($workQueaue)) \n"

#     $workQueaue[1,old]= $metaX
#     $workQueaue[2,old]= UInt16($metaY)
#     $workQueaue[3,old]= UInt16($metaZ)
#     $workQueaue[4,old]= UInt16($isGold)
#     end)#qote 
# end#appendToWorkQueue

end#WorkQueueUtils

# basicList = [-1,0,1]
# function isAdjacent(liist)::Bool
#     for tupl in liist
#         for tuplInner in liist
#             xx=0;yy=0;zz=1
#             if( (tupl[1]+xx,tupl[2]+yy,tupl[3]+zz)== tuplInner   )
#                     return true
#                 end

#            xx=0;yy=0;zz=-1
#             if( (tupl[1]+xx,tupl[2]+yy,tupl[3]+zz)== tuplInner   )
#                     return true
#                 end

#            xx=0;yy=-1;zz=0
#             if( (tupl[1]+xx,tupl[2]+yy,tupl[3]+zz)== tuplInner   )
#                     return true
#                 end

#            xx=0;yy=1;zz=0
#             if( (tupl[1]+xx,tupl[2]+yy,tupl[3]+zz)== tuplInner   )
#                     return true
#                 end

#            xx=1;yy=0;zz=0
#             if( (tupl[1]+xx,tupl[2]+yy,tupl[3]+zz)== tuplInner   )
#                     return true
#                 end

#            xx=-1;yy=0;zz=0
#             if( (tupl[1]+xx,tupl[2]+yy,tupl[3]+zz)== tuplInner   )
#                     return true
#                 end

#         end
#     end    
#     return false
# end


# even= [2,4,6]
# odd = [1,3,5]

# lA = []
# for x in even,y in even,z in even
#     push!(lA,(x,y,z)  )
# end
# lB = []
# for x in odd,y in even,z in even
#     push!(lB,(x,y,z)  )
# end
# lC = []
# for x in even,y in odd,z in even
#     push!(lC,(x,y,z)  )
# end
# lD = []
# for x in even,y in even,z in odd
#     push!(lD,(x,y,z)  )
# end
# lE = []
# for x in odd,y in odd,z in even
#     push!(lE,(x,y,z)  )
# end
# lF = []
# for x in even,y in odd,z in odd
#     push!(lF,(x,y,z)  )
# end
# lG = []
# for x in odd,y in odd,z in odd
#     push!(lG,(x,y,z)  )
# end
# lH = []
# for x in odd,y in even,z in odd
#     push!(lH,(x,y,z)  )
# end

# alll =[lA,lB,lC,lD,lE,lF,lG,lH] 


# map(it-> isAdjacent(it),alll)
# sum(map(it-> length(it),alll))== 6*6*6



# evens




    #@appendToWorkQueueBasic(workQueue,workQueuecounter, $metaX,$metaY,$metaZ,$isGold )

        # if(iseven($metaX) && iseven($metaY) && iseven($metaZ) )
        #     @appendToWorkQueueBasic(workQueueEEE,workQueueEEEcounter, $metaX,$metaY,$metaZ,$isGold )
        # elseif(iseven($metaX) && isodd($metaY) && iseven($metaZ))    
        #     @appendToWorkQueueBasic(workQueueEOE,workQueueEOEcounter, $metaX,$metaY,$metaZ,$isGold )
        # elseif(iseven($metaX) && iseven($metaY) && isodd($metaZ))    
        #     @appendToWorkQueueBasic(workQueueEEO,workQueueEEOcounter, $metaX,$metaY,$metaZ,$isGold )
        # elseif(isodd($metaX) && iseven($metaY) && iseven($metaZ))    
        #     @appendToWorkQueueBasic(workQueueOEE,workQueueOEEcounter, $metaX,$metaY,$metaZ,$isGold )
        # elseif(isodd($metaX) && isodd($metaY) && iseven($metaZ))    
        #     @appendToWorkQueueBasic(workQueueOOE,workQueueOOEcounter, $metaX,$metaY,$metaZ,$isGold )
        # elseif(iseven($metaX) && isodd($metaY) && isodd($metaZ))    
        #     @appendToWorkQueueBasic(workQueueEOO,workQueueEOOcounter, $metaX,$metaY,$metaZ,$isGold )
        # elseif(isodd($metaX) && iseven($metaY) && isodd($metaZ))    
        #     @appendToWorkQueueBasic(workQueueOEO,workQueueOEOcounter, $metaX,$metaY,$metaZ,$isGold )  
        # elseif(isodd($metaX) && isodd($metaY) && isodd($metaZ))    
        #     @appendToWorkQueueBasic(workQueueOOO,workQueueOOOcounter, $metaX,$metaY,$metaZ,$isGold )
        # end

            # workQueueEEE,workQueueEEEcounter,workQueueEEO,workQueueEEOcounter
    # ,workQueueEOE,workQueueEOEcounter,workQueueOEE,workQueueOEEcounter
    # ,workQueueOOE,workQueueOOEcounter,workQueueEOO,workQueueEOOcounter
    # ,workQueueOEO,workQueueOEOcounter,workQueueOOO,workQueueOOOcounter
    # ,CUDA.zeros(UInt16,4,(queueSize)),CUDA.zeros(Int16,(1)) 
    # ,CUDA.zeros(UInt16,4,(queueSize)),CUDA.zeros(Int16,(1)) 
    # ,CUDA.zeros(UInt16,4,(queueSize)),CUDA.zeros(Int16,(1)) 
    # ,CUDA.zeros(UInt16,4,(queueSize)),CUDA.zeros(Int16,(1)) 
    # ,CUDA.zeros(UInt16,4,(queueSize)),CUDA.zeros(Int16,(1)) 
    # ,CUDA.zeros(UInt16,4,(queueSize)),CUDA.zeros(Int16,(1)) 
    # ,CUDA.zeros(UInt16,4,(queueSize)),CUDA.zeros(Int16,(1))     