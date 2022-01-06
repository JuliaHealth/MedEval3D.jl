"""
based on  https://github.com/Visceral-Project/EvaluateSegmentation/blob/master/source/MutualInformationMetric.h
"""
module InformationTheorhetic
using ..BasicStructs, Parameters,  CUDA

"""
calculate mutual information  based on precalulated constants
"""
function mutualInformationMetr(tn,tp,fp, fn) ::Float64
    H1=0
    H2=0
    H12=0
    if(tn>0)
        # print("tn $(tn) tp $(tp) fp $(fp) fn $(fn)\n ")
        n = tn+tp+fp+fn
        # print("tn $(tn) tp $(tp) fp $(fp) fn $(fn) n $(n) \n ")
        row1 = tn + fn 
        row2 = fp + tp 
        # print("row1 $(row1) row2 $(row2) row1/n $(row1/n) row2/n $(row2/n)")

        H1 = - ( (row1/n)*log2(row1/n) + (row2/n)*log2(row2/n)) 

        col1 = tn + fp 
        col2 = fn + tp 
        H2 = - ( (col1/n)*log2(col1/n) + (col2/n)*log2(col2/n)) 

        p00::Float32 = tn==0 ? 1 : (tn/n) 
        p01::Float32 = fn==0 ? 1 : (fn/n) 
        p10::Float32 = fp==0 ? 1 : (fp/n) 
        p11::Float32 = tp==0 ? 1 : (tp/n) 
        H12= - ( (tn/n)*log2(p00) + (fn/n)*log2(p01) +  (fp/n)*log2(p10) + (tp/n)*log2(p11) ) 
    end    
return H1+H2-H12 

end #calculateVolumeMetric

"""
The Variation of Information (VOI) measures the
amount of information lost (or gained) when changing
from one variable to the other
"""
function variationOfInformation(tn,tp,fp, fn)
    H1=0
    H2=0
    Mi=0
    H12=0
    n = tn+tp+fp+fn
    trueVoxels = fn + tp;
    retrievedVoxels = fp + tp;
    if(tn>0)

    H1 = - ( (trueVoxels/n)*log2(trueVoxels/n) + (1- trueVoxels/n)*log2(1- trueVoxels/n));
    H2 = - ( (retrievedVoxels/n)*log2(retrievedVoxels/n) + (1- retrievedVoxels/n)*log2(1- retrievedVoxels/n) );
    p00::Float32 = tn==0 ? 1.0 : (tn/n);
     p01::Float32 = fn==0 ? 1.0 : (fn/n);
     p10::Float32 = fp==0 ? 1.0 : (fp/n);
     p11::Float32 = tp==0 ? 1.0 : (tp/n);
   H12= - ( (tn/n)*log2(p00) 
                + (fn/n)*log2(p01) +  (fp/n)*log2(p10) + (tp/n)*log2(p11) )
    end
    MI=H1+H2-H12;
    
    return H1+H2-2*MI;


end#variationOfInformation

end #InformationTheorhetic