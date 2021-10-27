


"""
utility functions helping managing result list 
"""
module ResultListUtils
using CUDA
export getResLinIndex,allocateResultList

"""
allocate memory on GPU for storing result list 
    totalFpCount-  total number of false positives
    TotalFNCount - total number of false negatives

    in the array first 3 entries will be x,y,z than isGold - 1 if it is  related to gold pass dilatations
        , direction from which result was set and the iteration number in which it was covered
"""
function allocateResultList(totalFpCount,TotalFNCount)
return CUDA.zeros(UInt16, (totalFpCount+ TotalFNCount+1),6 )
end#allocateResultList

"""
giver the result row that holds data about covered point and in what iteration, from what direction and in what pass it was covered
resRow - array where first 3 entries are x,y,z positions then is gold,
"""
 function getResLinIndex(resRow,mainArrDims)
    # last one is in order to differentiate between gold pass and other pass ...
    return resRow[1]+ resRow[2]*mainArrDims[1]+ resRow[3]* mainArrDims[1]*mainArrDims[2]+ resRow[4]*mainArrDims[1]*mainArrDims[2]*mainArrDims[1]*mainArrDims[3]  
 end

end#ResultListUtils