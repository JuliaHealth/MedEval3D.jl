module BasicStructs
using Parameters
export ResultMetrics,ConfigurtationStruct
"""
constants associated with image over which we will evaluate segmentations
"""
@with_kw struct ImageConstants
    mvspx ::Float64 # Voxelspacing x 
    mvspy::Float64 # Voxelspacing y
    mvspz::Float64 #mean Voxelspacing z
    isZConst::Bool # true if slices thickness is the same in all image
    ZPositions::Vector{Float64} # array of true physical positions (in mm) of slices relative to the begining - used in case we have variable thickness of slices
    numberOfVox::Int64 # number of voxels in image
end #ImageConstants
"""
configuration struct that when passed will marks what kind of metrics we are intrested in 
    
"""
@with_kw struct ConfigurtationStruct
    sliceWiseMatrics::Bool = false# if it will be marked as true metrics will be calculated not only globally but also 
    dice::Bool = false #dice coefficient
    jaccard::Bool = false #jaccard coefficient
    gce::Bool = false #global consistency error
    vol::Bool = false# Volume metric
    randInd::Bool= false # Rand Index 
    ic::Bool= false # interclass correlation
    kc::Bool= false # Kohen Cappa
    mi::Bool= false # mutual information
    vi::Bool= false # variation Of Information
    md::Bool= false # mahalanobis distance
    hd::Bool= false # hausdorff distance
end #ConfigurtationStruct


"""
Struct holding all resulting metrics - if some metric was not calculated its value is just -1  
"""
@with_kw struct ResultMetrics
    dice::Float64 = -1.0 #dice coefficient
    jaccard::Float64 =  -1.0 #jaccard coefficient
    gce::Float64 =  -1.0 #global consistency error
    vol::Float64 =  -1.0 # Volume metric
    randInd::Float64 = -1.0 # Rand Index 
    ic::Float64 = -1.0 # interclass correlation
    kc::Float64 = -1.0 # Kohen Cappa
    mi::Float64 = -1.0 # mutual information
    vi::Float64 = -1.0 # variation Of Information
    md::Float64 = -1.0 # mahalanobis distance
    hd::Float64 = -1.0 # hausdorff distance

end #ResultMetrics

# """
# struct holding necess
# TN- true negative   TP - true positive  FN - false negative  FP - false positive
# """
# @with_kw struct TnTpFpFn 
# # look into https://github.com/JuliaArrays/StructArrays.jl

# end #TnTpFpFn



end#BasicStructs
