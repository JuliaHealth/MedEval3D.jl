module WorkQueueUtils
using  CUDA,Main.CUDAAtomicUtils
export allocateWorkQueue,appendToWorkQueue
"""
allocate memory for  work queues
    entries means
        1)xMeta
        2)yMeta
        3)zMeta
        4)isGold
"""
function allocateWorkQueue(fpTotal,fnTotal)
    return CUDA.zeros(UInt8,4,Int64(ceil((fpTotal+fnTotal +1)*1.51)))
end

"""
atomically append the block linear index and information is it gold or other pass 
"""
function appendToWorkQueue(workQueaue,workQueauecounter, metaX,metaY,metaZ,isGold ) 
   old =  atomicallyAddOne(workQueauecounter)+1
  # CUDA.@cuprint "in appendToWorkQueue metaX $(metaX) metaY $(metaY) metaZ $(metaZ) isGold $(isGold) old $(old) \n"

   workQueaue[1,old]= UInt8(metaX)
   workQueaue[2,old]= UInt8(metaY)
   workQueaue[3,old]= UInt8(metaZ)
   workQueaue[4,old]= UInt8(isGold)

end#appendToWorkQueue

end#WorkQueueUtils