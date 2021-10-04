"""
It holds macos and functions usefull for reductions of 3 dimensional data 

"""
module ReductionUtils

"""
w will supply here some thread local variable, then this variable will be reduced between lanes
hence operation will be first applied  to the threads in a warp 
- next the first thred in each warp will have reduced value and it will pass it to the shared memory
- next we will reduce the value in shared memory using informationa passing in warps - what is important as at this step only 1warp is utilized we will 
  try to parallelize and do the final reduction of each variable on separate warp
!! important it is assumed that we have more warps at our diposal than amount of variables we are reducing 
all of those functions are supposed to be invoked from inside of the kernel
"""
macro redWitAct()
  
end#reduceWitAction


"""
will be used to send reduced value atomically to some array  (ussually in global memory)
"""
macro sendAtomic()
  
end#sendAtomic

"""
given counter and array it will atomically inrease counter value and on the basis of old will 
push given value into the array at this spot - so to the end of array - thanks to atomic usage of counter we eill not overwrite data 
"""
macro append()
  
end


"""
modified append   so we want to append whole array at once - ussually we will have some array in shared memory and when it will get filled 
  we will want to pass it into global memory so we need 
  globalArray - the taret in global memory
  localArray - ussually in shared memory  - used to hold some results before bulk send
  globalArraycounter - accessed atomically keeps information about where is the end of array
localArrayCounter - accessed atomically keeps information where is the end of local array

so we will
  1) write results to the local array
  2) when finished loop or when array is filled we will  increse the gloal counter by the local counter
      local counter needs to be set to 0 atomically - old value will be used to increase global counter as described above
  3) we will write results to global starting from old global counter spot
 
"""
macro appendMulti()
  
end


end #module ReductionUtils
