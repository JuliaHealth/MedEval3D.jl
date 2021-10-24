module WorkQueueUtils
using  CUDA,Main.CUDAAtomicUtils
"""
allocate memory for  work queues
"""
function allocateWork_Fp_Fn_Queues(fpTotal,fnTotal)
    return CUDA.zeros(UInt8,Int64(ceil((fpTotal+fnTotal )*1.51)),4)
end

"""
atomically append the block linear index and information is it gold or other pass 
"""
function appendToWorkQueue(workQueaue,workQueauecounter, metaX,metaY,metaZ,isGold ) 
   old =  atomicallyAddOne(workQueauecounter)+1
   workQueaue[old,1]= UInt8(metaX)
   workQueaue[old,2]= UInt8(metaY)
   workQueaue[old,3]= UInt8(metaZ)
   workQueaue[old,4]= UInt8(isGold)

end#appendToWorkQueue

end#WorkQueueUtils