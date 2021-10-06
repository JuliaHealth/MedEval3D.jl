"""
It holds macos and functions usefull for reductions of 3 dimensional data 

"""
module ReductionUtils
using Main.CUDAGpuUtils
export @redWitAct
"""
adapted from https://discourse.julialang.org/t/macro-magic-looping-over-varargs-printing-values-and-symbols/3025

w will supply here some thread local variable, then this variable will be reduced between lanes
hence operation will be first applied  to the threads in a warp 
- next the first thred in each warp will have reduced value and it will pass it to the shared memory
- next we will reduce the value in shared memory using informationa passing in warps - what is important as at this step only 1warp is utilized we will 
  try to parallelize and do the final reduction of each variable on separate warp
!! important it is assumed that we have more warps at our diposal than amount of variables we are reducing 
all of those functions are supposed to be invoked from inside of the kernel
varActTuples - is a list of tuples where first in tuple is variable on which we want to do reduction operation
    and second is the operation for example +

so in the end we will have fully reducd values in shared memory in first entries of the x dimension of shared memory for each variable analyzed
  so it is basically mapreduce operation but implemented inside the block - and basically block private map reduce
!!!!!!!! important we assume for some standarization that we have thread block of x dimension 32
    and y dimension at least as big as number of variables we pass to reduce 

    offsetIter - we need to use offset number that we will shift - in order to prevent creating new variable
  we can reuse some alredy created earlier or create a new one 
  shmem - shared memory used for reduction it needs to have length of 32 and number of rows (y dim ) at least as big as number of variables reduced
  """
macro redWitAct(offsetIter,shmem, varActTuples...)

    # tmp = []
    # push!(tmp, quote $while($offsetIter <32)  end)
    # # e1 = Expr(:call, while , 3, 4)
    # for i in 0:cld(length(varActTuples),2)-1
    #     offSet = i*2
    #     el1 = varActTuples[offSet+1]
    #     op = varActTuples[offSet+2]
    #     #push!(tmp, :(print($op($el1, 2 ))   ))
    #     push!(tmp, :(@inbounds $el1=$op($el1,shfl_down_sync(FULL_MASK, $el1, offsetIter))  ))
    # end#for

    # push!(tmp, :(end))

return esc(quote
  while($offsetIter <32) 
    $offsetIter<<= 1
  end

end)
# return esc(quote
#   while($offsetIter <32) 
#     Expr(:block, $tmp...)
#     $offsetIter<<= 1
#   end

# end)
    # return esc(
    # tmp[1]
  
    #  )
    # while(offsetIter <32) 
    #   @inbounds sumX+=shfl_down_sync(FULL_MASK, sumX, offsetIter)  
    # offsetIter<<= 1
    #   end
    #   if(threadIdxX()==1)
    #   @inbounds shmemSum[threadIdxY(),1]+=sumX
    #   @inbounds shmemSum[threadIdxY(),2]+=sumY
    #   @inbounds shmemSum[threadIdxY(),3]+=sumZ
    #   @inbounds shmemSum[threadIdxY(),4]+=count
    #   end
    #   sync_threads()




end#reduceWitAction


"""
will be used to send reduced value atomically to some array  (ussually in global memory)
so we will take as arguments shared memory and in given order varargs arrays to which we want to add  values from shared memory
so we will add atomically to the first variable shmemSum[1,1] to second shmemSum[1,2] etc ...
we will check also is the value we want to send is not zero - if it is we will not send it to global variable 
!!!!!! important we assume we will ave at least as many warps as variables we want to send
"""
macro sendAtomic(shmemSum, vars)

  tmp = []
  index = 0
  for varr in vars
    index+=1
      push!(tmp, :( @ifXY $(index) $(index) if(shmemSum[1,$(index)]>0)   @inbounds @atomic $varr[]+= shmemSum[1,$(index)] end    ))
  end#for

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
