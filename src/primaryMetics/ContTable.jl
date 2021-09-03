using DrWatson
@quickactivate "Medical segmentation evaluation"
"""
module based on https://github.com/Visceral-Project/EvaluateSegmentation/blob/master/source/ContingencyTable.h
"""
module ContTable
using Main.BasicStructs, Parameters, Setfield

"""
giving set of constant associated with  the image that will be usefull for calculating 
    I - image the we are trying to segment - 3 dimensional array of supplied type imageNumb
    return ImageConstants - some constants related to image iself (nor masks)
    """
function getImageConstants(::Type{imageNumb} 
                      I::Array{imageNumb, 3})
                      ::ImageConstants
                      where{imageNumb}


  return ImageConstants(
            mvspx = # Voxelspacing x 
            mvspy= # Voxelspacing y
            mvspz= #mean Voxelspacing z
            isZConst = # true if slices thickness is the same in all image
            numberOfVox= # number of voxels in image

  )

end #getImageConstants


"""
calculating tn, fn, fp and tp WITHOUT voxel volume correction
Type{maskNumb}  - type of the numbers hold in mask
G - 3 dimensional array holding ground truth segmentation
T - 3 dimensional array holding segmentation that we want to compare to ground truth
isVariedSlice- true if slices can have varying thickness
imageConst - image dependent constants
return array of doubles with entry 1)TN- true negative 2)  TP - true positive 3) FN - false negative  4) FP - false positive
"""
function getTnTpFpFn(::Type{maskNumb} 
                ,G::Array{maskNumb, 3}
                 ,T::Array{maskNumb, 3}
                  ,isVariedSlice::Bool
                  ,imageConst::ImageConstants
    )::Vector{Float64}
     where{maskNumb}
    
    x1::Float64 =((double)values_f[i])/((double)PIXEL_VALUE_RANGE_MAX)
    y1::Float64  =1-x1
    x2::Float64 =1-x2
    tn += min(y1,y2)
    tp += min(x1,x2)
    fn += x1>x2 ? x1-x2 : 0
    fp += x2>x1 ? x2-x1 : 0
    
	look inot https://github.com/JuliaArrays/StructArrays.jl
	
return [tn,tp,fp,fn ] 
end   #TnTpFpFn





"""
calculating  a,b,c,d constants - needed for some particulary pairwise comparison metrics WITHOUT voxel volume correction
isVariedSlice- true if slices can have varying thickness
imageConst - image dependent constants
TnTpFpFns - calculated Tn Tp Fp and Fn in that order 
return array of doubles with entry 1) a  2)  b 3) c  4) FP d 
"""

function getAbcdConsts(isVariedSlice::Bool
                  ,imageConst::ImageConstants
                  ,TnTpFnFps::Vector{Bool}
    )::Vector{Float64}
	n = std::min(numberElements_f, numberElements_m);
	coltot1 = tn + fp;
	coltot2 = fn + tp;
	rowtot1 = tn + fn;
	rowtot2 = fp + tp;
    nis = rowtot1*rowtot1 + rowtot2*rowtot2;  
    njs = coltot1*coltot1 + coltot2*coltot2;  
    s = tp*tp + tn*tn + fp*fp + fn*fn ;
    
    a = ( binomial(tn) + binomial(fn) + binomial(fp) + binomial(tp) )/2.0;
    
    b = (njs - s)/2.0;
    c = (nis - s)/2.0;
    d = ( n*n + s - nis - njs )/2.0;     

return [a,b,c,d]

end   #getAbcdConsts


"""
helper function calculating binomial
"""
function binomial(val::Float64)::Float64
    return  val*(val-1)
end #binomial



end #ContTable
