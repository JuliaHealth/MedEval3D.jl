"""
makes some bitwise operations easier
"""

module BitWiseUtils

"""
sets given bit of supplied number to 1
"""
macro setBitTo1(numb,position)
    return esc(quote

    end)
  
end#setBitTo1
"""
set bit of number numb in position pos to value val
"""
macro setBitTo(numb,pos,val)
    return esc(quote

    end)
  
end#setBitTo1
setBitTo


end#BitWiseUtils
