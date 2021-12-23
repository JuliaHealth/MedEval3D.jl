
module CUDAAtomicUtils
using CUDA 
using CUDA: @atomic

export atomicallyAddOne,atomicallyAddToSpot, atomicAdd, atomicMinSet, atomicMaxSet
"""
atomically add to given 1 length array 1
"""
function atomicallyAddOne(arr) 
   return  CUDA.@atomic arr[1]+=1
end


"""
atomically add given value to the coordinate (linear) of supplied array
"""
function atomicallyAddToSpot( arr,coord,value)
#    return CUDA.atomic_add!(pointer(arr, coord),typ(value))
return  CUDA.@atomic arr[coord]+=value
end

"""
adds atomically number to target and return old value
"""
function atomicAdd(target, number)
    return  CUDA.@atomic target[1]+=number

end

"""
given array (or some pointer ...)  and value to set it will set target to miniumum of targets value and supplied value and return old value
target - what we want to updated
value - new value if is the minimum
coord - coordinate of a pointer in array
"""
function atomicMinSet( arr,value,coord=1)
   
    #return CUDA.atomic_add!(pointer(arr, coord),typ(value))
  return  @inbounds CUDA.@atomic arr[coord]=min(arr[coord],value)
end#atomicMinSet


"""
given array (or some pointer ...)  and value to set it will set target to maximum of targets value and supplied value  and return old value
"""
function atomicMaxSet( arr,value,coord=1)
    return @inbounds CUDA.@atomic arr[coord]=max(arr[coord],value)

end#atomicMaxSet





 end#module CUDAAtomicUtils