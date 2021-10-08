"""
It holds macos and functions usefull for reductions of 3 dimensional data 

"""
module ReductionUtils
using Main.CUDAGpuUtils, CUDA
export @redWitAct, @addAtomic,@addNonAtomic, @redOnlyStepOne,@redOnlyStepThree
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

firstPart =   reduceWitActFirstPart(offsetIter,shmem, varActTuples...)
secondPart =   reduceWitActSecondPart(offsetIter,shmem, varActTuples...)
thirdPart =   reduceWitActThirdPart(offsetIter,shmem, varActTuples...)

  return esc(:(
    $offsetIter=1;  
  while($offsetIter <32) 
        $firstPart
        $offsetIter<<= 1
    end;
    if(threadIdxX()==1)
    $secondPart
    end;
    sync_threads();
    $offsetIter=1;
    $thirdPart
    ))

end#reduceWitAction

"""
modification of redWitAct - where we reduce only across the warp and save it to shmem
- so at the end of this step
      we will get reduced values from each of the warp in the first lane of the warp
"""
macro redOnlyStepOne(offsetIter,shmem, varActTuples...)
  firstPart =   reduceWitActFirstPart(offsetIter,shmem, varActTuples...)

  
    return esc(:(
      $offsetIter=1;  
    while($offsetIter <32) 
          $firstPart
          $offsetIter<<= 1
      end;

          ))
  
  end#reduceWitAction


  """
  modification of redWitAct - where we reduce only across shared memory
   """
  macro redOnlyStepThree(offsetIter,shmem, actions...)
  
    thirdPart =   reduceWitActThirdPartOnly(offsetIter,shmem, actions)

      return esc(:(
        $offsetIter=1;
        $thirdPart
            ))
    
    end#reduceWitAction



"""
First stage of reductions where local variables are added from registers to registers
"""
function reduceWitActFirstPart(offsetIter,shmem, varActTuples...)
  tmp = []
  for i in 0:cld(length(varActTuples),2)-1
      offSet = i*2
      el1 = varActTuples[offSet+1]
      op = varActTuples[offSet+2]
      push!(tmp, :(@inbounds $el1=$op($el1,shfl_down_sync(FULL_MASK, $el1, $offsetIter))  ))
  end#for
  return Expr(:block,tmp...)
end
"""
second stage of reduction where set the values from the first lane in warp to shared memory
important I keep the convention to have 2 dimensional thread blocks where x dimension is 32 
- it makes warp opertations simpler
"""
function reduceWitActSecondPart(offsetIter,shmem, varActTuples...)
  tmp = []
  index = 0
  for i in 0:cld(length(varActTuples),2)-1
      index+=1
      offSet = i*2
      el1 = varActTuples[offSet+1]
      op = varActTuples[offSet+2]
      push!(tmp, :(@inbounds $shmem[threadIdxY(),$index]= $el1))
  end#for
  return Expr(:block,tmp...)
end

"""
third stage of reduction in block 
we will get variables in the same order as was passed in primary macro 
at this step those are already in the shared memory 
    - number of those is the same as number of warps
 using warp reduction we will reduce information from shared memory
 in order to pralelize the process we will use separate warp for each operation   

"""
function reduceWitActThirdPart(offsetIter,shmem, varActTuples...)
  tmp = []
  index = 0
  for i in 0:cld(length(varActTuples),2)-1
      index+=1
      offSet = i*2
      el1 = varActTuples[offSet+1]
      op = varActTuples[offSet+2]
      push!(tmp, quote
      if(threadIdxY()==$index)
        while(offsetIter <32) 
          @inbounds $shmem[threadIdxX(),$index]=$op($shmem[threadIdxX(),$index],  shfl_down_sync(FULL_MASK, shmemSum[threadIdxX(),$index], $offsetIter))  
          offsetIter<<= 1
        end
      end  
      end)
  end#for
  return Expr(:block,tmp...)
end
"""
modification where we pass only actions - as we know from indicies what need to be done
"""
function reduceWitActThirdPartOnly(offsetIter,shmem, actions)
  tmp = []
  for index in 1:length(actions)
      op = actions[index]
      push!(tmp, quote
      if(threadIdxY()==$index)
        while(offsetIter <32) 
          @inbounds $shmem[threadIdxX(),$index]=$op($shmem[threadIdxX(),$index],  shfl_down_sync(FULL_MASK, shmemSum[threadIdxX(),$index], $offsetIter))  
          offsetIter<<= 1
        end
      end  
      end)
  end#for
  return Expr(:block,tmp...)
end



"""
will be used to send reduced value atomically to some array  (ussually in global memory)
so we will take as arguments shared memory and in given order varargs arrays to which we want to add  values from shared memory
so we will add atomically to the first variable shmemSum[1,1] to second shmemSum[1,2] etc ...
we will check also is the value we want to send is not zero - if it is we will not send it to global variable 
!!!!!! important we assume we will ave at least as many warps as variables we want to send
"""
macro addAtomic(shmemSum, vars...)

  mainPArt = sendAtomicHelperAndAdd(shmemSum, vars...)

  return esc(:($mainPArt  ))

end#sendAtomic

"""
we get information about reduction  from shared memory - we know which entry from order we get the
data in vars  - are variables - ussually in global memory to which we want to push the value
"""
function sendAtomicHelperAndAdd(shmemSum, vars...)
  tmp = []
  for index in 1:length(vars)
      varr = vars[index]
      push!(tmp, quote
      @ifXY $index $index if(shmemSum[1,$index]>0)   @inbounds @atomic $varr[]+=$shmemSum[1,$(index)]  end   
       
      end)
  end#for
  return Expr(:block,tmp...)



  for index in 1:length(varActTuples)
    varr= vars[index]
      push!(tmp, quote
     @ifXY $index $index if(shmemSum[1,$index]>0)   @inbounds @atomic $varr[]= $shmemSum[1,$(index)] end   
  
      end)
  end#for
  return Expr(:block,tmp...)
end







"""
modification of above where we do the addition in nonatomic fashion
"""
macro addNonAtomic(shmemSum, vars...)

  mainPArt = sendNonAtomicHelperAndAdd(shmemSum, vars...)

  return esc(:($mainPArt  ))

end#sendAtomic

"""
we get information about reduction  from shared memory - we know which entry from order we get the
data in vars  - are variables - ussually in global memory to which we want to push the value
"""
function sendNonAtomicHelperAndAdd(shmemSum, vars...)
  tmp = []
  for index in 1:length(vars)
      varr = vars[index]
      push!(tmp, quote
      @ifXY $index $index if(shmemSum[1,$index]>0)   @inbounds $varr[1]+=$shmemSum[1,$(index)]  end   
       
      end)
  end#for
  return Expr(:block,tmp...)



  for index in 1:length(varActTuples)
    varr= vars[index]
      push!(tmp, quote
     @ifXY $index $index if(shmemSum[1,$index]>0)   @inbounds @atomic $varr[]= $shmemSum[1,$(index)] end   
  
      end)
  end#for
  return Expr(:block,tmp...)
end










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
