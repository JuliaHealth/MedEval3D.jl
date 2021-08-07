using DrWatson
@quickactivate "Medical segmentation evaluation"
module GPUutils

defineBlocksStr= """
Type{maskNumb}  - type of the numbers hold in mask
G - 3 dimensional array holding ground truth segmentation
T - 3 dimensional array holding segmentation that we want to compare to ground truth
isVariedSlice - if true it will mark that  slices have varying thickness  - hence we will need later corrextion...

"""
@doc defineBlocksStr
function defineBlocks(::Type{maskNumb} 
                    ,G::Array{maskNumb, 3}
                    ,T::Array{maskNumb, 3}
                    ,isVariedSlice::Bool
                    ) where{maskNumb}


end#defineBlocks



end #GPUutils