
"""
based on  https://github.com/Visceral-Project/EvaluateSegmentation/blob/master/source/VolumeSimilarityCoefficient.h

"""
module VolumeMetric
using Main.BasicStructs, Parameters, Setfield


"""
calculate volume metric  based on precalulated constants
TnTpFpFns - list of basic metrics - in the order [tn,tp,fp,fn ] 
res - ResultMetrics struct holding result of all metrics calculated for given run
return modified ResultMetrics with added volume metric
"""
function calculateVolumeMetric(TnTpFpFns::Vector{Float64} , res::ResultMetrics) ::ResultMetrics
    tn = TnTpFpFns[1]
    tp = TnTpFpFns[2]
    fp = TnTpFpFns[3]
    fn = TnTpFpFns[4]
return setproperties(res, (vol= getVolumMetric(tp,fp, fn)  )) 

end #calculateVolumeMetric


"""
calculates volume metric
"""
function getVolumMetric(tp,fp, fn )
    return 1- abs(fn-fp)/(2*tp + fn + fp) 
end#getVolumMetric


end#VolumeMetric