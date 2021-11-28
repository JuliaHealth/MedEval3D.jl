"""
makes some bitwise operations easier
"""

module BitWiseUtils

export @setBitTo1, @setBitTo , isBit1AtPos, @bitDilatate,@bitPassOnes
"""
sets given bit of supplied number to 1
"""
macro setBitTo1(numb,pos)
    return esc(quote
        $numb|= 1 << ($pos-1)
    end)
  
end#setBitTo1
"""
set bit of number numb in position pos to value val
"""
macro setBitTo(numb,pos,val)
    return esc(quote
    # Suppose you want to change bit N of x, where N=0 means the least-significant bit. You don’t care what the old value was, but the new value is B (either 0 or 1). This will do the trick:
     ($numb) =(($numb) & ~(1<<($pos-1))) | ($(val)<<($pos-1))
    end)
  
end#setBitTo1

"""
returning boolean indicating whether we have 1 at given spot in the integer
"""
function isBit1AtPos(numb,pos)::Bool
  return  (numb & (1 << (pos - 1)))>0
end#isBit1AtPos
    

"""
needed for bit dilatation in the kernel 
    Given 32 bit integer source and target i woul like to do sth like one dimensional mathematical morphology dilatation so given bit at position x is 1 in source i would like to set bits at position x+1 and x-1 at target to 1 Other bits of target should not be changed so for example

Given 1 0 0 0 1 0 0 0 ... Set target integer into 1 1 0 1 1 1 0 0 ...

https://stackoverflow.com/questions/70134566/bit-wise-dilatation
"""
macro bitDilatate(x)
    return esc(quote
      (($x )>> 1) | ($x) | (($x) << 1)
    end)#quote
end

"""
Given 32 bit integers x  and y  i would like to set bits of x to 1 if in corresponding position of y there is 1  without modyfing other bits of y 
    So for example  if x is 1 0 0 0 0 1 ...And y is 0 0 0 1 0 0  ...
    I would like a result that would be  1 0 0 1 0 1 
"""
macro bitPassOnes(source,target)
    return esc(quote
         (($target)|($source))
    end)#quote
end


end#BitWiseUtils


# for i in 1:16
#     div,remm = divrem(i-1,4)
#     # aFrag[fld(div+1,2)+1,rem+1 ] = dataShmem[rem+1,fld(div+1,2)+1 ]

#     a =  ((i-1) & (3))+1
#     b = ~((i+3)>>2 & 1)+3
#     c = ((i-1)>>2 )+1
#     aFrag[a,c] = dataShmem[b,a]*( ((i>4 && i<13)*-2)+1 )
# @info "i$(i)   a $(a)   b $(b)   c $(c)"

# end 


# numb = UInt32(0)
# #settingcorrectly
# numb |= 1 << 1
# numb |= 1 << 2
# numb |= 1 << 0
# numb |= 1 << 0
# numb |= 1 << 5
# numb


#reading...
# numb>>1 & UInt32(1) 
# numb>>2 & UInt32(1) 
# numb>>3 & UInt32(1) 
# numb>>4 & UInt32(1) 
# numb>>5 & UInt32(1) 