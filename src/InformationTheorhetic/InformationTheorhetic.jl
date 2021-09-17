using DrWatson
@quickactivate "Medical segmentation evaluation"
"""
based on  https://github.com/Visceral-Project/EvaluateSegmentation/blob/master/source/MutualInformationMetric.h
"""
module InformationTheorhetic
using Main.BasicStructs, Parameters, Setfield

"""
calculate mutual information  based on precalulated constants
"""
function mutualInformationMetr(tn,tp,fp, fn) ::Float64
    n = tn+tp+fp+fn
    row1 = tn + fn 
    row2 = fp + tp 
    H1 = - ( (row1/n)*log2(row1/n) + (row2/n)*log2(row2/n)) 

    col1 = tn + fp 
    col2 = fn + tp 
    H2 = - ( (col1/n)*log2(col1/n) + (col2/n)*log2(col2/n)) 

    p00 = tn==0 ? 1 : (tn/n) 
    p01 = fn==0 ? 1 : (fn/n) 
    p10 = fp==0 ? 1 : (fp/n) 
    p11 = tp==0 ? 1 : (tp/n) 
    H12= - ( (tn/n)*log2(p00) + (fn/n)*log2(p01) +  (fp/n)*log2(p10) + (tp/n)*log2(p11) ) 

return H1+H2-H12 

end #calculateVolumeMetric

"""
The Variation of Information (VOI) measures the
amount of information lost (or gained) when changing
from one variable to the other
"""
function variationOfInformation(tn,tp,fp, fn)::Float64
    n = tn+tp+fp+fn
    trueVoxels = fn + tp;
    retrievedVoxels = fp + tp;
    
     H1 = - ( (trueVoxels/n)*log2(trueVoxels/n) + (1- trueVoxels/n)*log2(1- trueVoxels/n));
     H2 = - ( (retrievedVoxels/n)*log2(retrievedVoxels/n) + (1- retrievedVoxels/n)*log2(1- retrievedVoxels/n) );
     p00 = tn==0 ? 1 : (tn/n);
     p01 = fn==0 ? 1 : (fn/n);
     p10 = fp==0 ? 1 : (fp/n);
     p11 = tp==0 ? 1 : (tp/n);
    H12= - ( (tn/n)*log2(p00) + (fn/n)*log2(p01) +  (fp/n)*log2(p10) + (tp/n)*log2(p11) );
    
    MI=H1+H2-H12;
    
    return H1+H2-2*MI;



end#variationOfInformation

end #InformationTheorhetic