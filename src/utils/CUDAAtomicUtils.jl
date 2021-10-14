atomicMinSet
atomicMaxSet

"""
atomically add to given 1 length array 1
data type need to be Int32
"""
function atomicallyAddOneInt(arr)
    @inbounds @atomic arr[]+=Int32(1)
   # CUDA.atomic_inc!(pointer(arr), Int32(1))
end

"""
looking in arr for entry x,y,z and setting atomically value it return old one also
"""
function atomicallySetValueTrreeDim(arr,x,y,z,value)
    #@inbounds @atomic arr[]+=Int32(1)
    CUDA.atomic_xchg!(pointer(arr[x,y,z]), UInt32(value))
    #@inbounds @atomic arr[x,y,z]=UInt32(value)
end


"""
adds atomically number to target and return old value
"""
function atomicAdd(target, number)

end