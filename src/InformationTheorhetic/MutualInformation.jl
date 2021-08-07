using DrWatson
@quickactivate "Medical segmentation evaluation"
"""
based on  https://github.com/Visceral-Project/EvaluateSegmentation/blob/master/source/MutualInformationMetric.h
"""
module MutualInformation
using Main.BasicStructs, Parameters, Setfield

mutualInformationMetrStr = """
calculate mutual information  based on precalulated constants
TnTpFpFns - list of basic metrics - in the order [tn,tp,fp,fn ] 
res - ResultMetrics struct holding result of all metrics calculated for given run
imageConsts - constants associated with main image
return modified ResultMetrics with added mutual information
"""
@doc mutualInformationMetrStr
function mutualInformationMetr(TnTpFpFns::Vector{Float64} , res::ResultMetrics, imageConsts::ImageConstants) ::ResultMetrics
    tn = TnTpFpFns[1]
    tp = TnTpFpFns[2]
    fp = TnTpFpFns[3]
    fn = TnTpFpFns[4]
    n = imageConsts.numberOfVox
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
    MI=H1+H2-H12 

return setproperties(res, (mi=  MI)) 

end #calculateVolumeMetric

end #MutualInformation