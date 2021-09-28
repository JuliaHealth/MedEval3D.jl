"""
based on  https://github.com/Visceral-Project/EvaluateSegmentation/blob/master/source/MutualInformationMetric.h
"""
module InformationTheorhetic
using Main.BasicStructs, Parameters, Setfield, CUDA

"""
calculate mutual information  based on precalulated constants
"""
function mutualInformationMetr(tn,tp,fp, fn) ::Float64
    n = tn+tp+fp+fn
    row1 = tn + fn 
    row2 = fp + tp 
    H1 = - ( (row1/n)*CUDA.log2(row1/n) + (row2/n)*CUDA.log2(row2/n)) 

    col1 = tn + fp 
    col2 = fn + tp 
    H2 = - ( (col1/n)*CUDA.log2(col1/n) + (col2/n)*CUDA.log2(col2/n)) 

    p00::Float32 = tn==0 ? 1 : (tn/n) 
    p01::Float32 = fn==0 ? 1 : (fn/n) 
    p10::Float32 = fp==0 ? 1 : (fp/n) 
    p11::Float32 = tp==0 ? 1 : (tp/n) 
    H12= - ( (tn/n)*CUDA.log2(p00) + (fn/n)*CUDA.log2(p01) +  (fp/n)*CUDA.log2(p10) + (tp/n)*CUDA.log2(p11) ) 

return H1+H2-H12 

end #calculateVolumeMetric

"""
The Variation of Information (VOI) measures the
amount of information lost (or gained) when changing
from one variable to the other
"""
function variationOfInformation(tn,tp,fp, fn)
    n = tn+tp+fp+fn
    trueVoxels = fn + tp;
    retrievedVoxels = fp + tp;
    
    H1 = - ( (trueVoxels/n)*CUDA.log2(trueVoxels/n) + (1- trueVoxels/n)*CUDA.log2(1- trueVoxels/n));
    H2 = - ( (retrievedVoxels/n)*CUDA.log2(retrievedVoxels/n) + (1- retrievedVoxels/n)*CUDA.log2(1- retrievedVoxels/n) );
    p00::Float32 = tn==0 ? 1.0 : (tn/n);
     p01::Float32 = fn==0 ? 1.0 : (fn/n);
     p10::Float32 = fp==0 ? 1.0 : (fp/n);
     p11::Float32 = tp==0 ? 1.0 : (tp/n);
   H12= - ( (tn/n)*CUDA.log2(p00) 
                + (fn/n)*CUDA.log2(p01) +  (fp/n)*CUDA.log2(p10) + (tp/n)*CUDA.log2(p11) )
    
    MI=H1+H2-H12;
    
    return H1+H2-2*MI;


end#variationOfInformation

end #InformationTheorhetic