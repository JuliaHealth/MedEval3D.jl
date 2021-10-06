
"""
some utilities for managing memory
"""
module MemoryUtils
using Main.CUDAGpuUtils, CUDA
export clearSharedMemWarpLong
"""
In case we have 32 length shared memory we can clear it using given warps
o x dim is assumed to be from 1 to 32 and ydim is supplied
"""
function clearSharedMemWarpLong(shmem, ydim::UInt8 )
@unroll for i::UInt8 in UInt8(1):UInt8(ydim)
    @ifY i shmem[threadIdxX(),i]=UInt32(0)
end#for 
sync_threads()
end#clearSharedMemWarpLong


end#MemoryUtils