
using DrWatson
@quickactivate "Medical segmentation evaluation"

module ProbabilisticMetrics
using Main.BasicStructs, Parameters, Setfield

"""
calculate  probabilistic metrics - Intercalss Correlation and Cohen cappa  based on precalulated constants
G,T - ground truth and evaluated masks - 3 dimensional arrays of supplied type-  maskNumb
isVariedSlice - true if slices are of variable thickness
TnTpFpFns - list of basic metrics - in the order [tn,tp,fp,fn ] 
res - ResultMetrics struct holding result of all metrics calculated for given run
return modified ResultMetrics with added  Intercalss Correlation and Cohen cappa
"""
calculateProbabilisticMetricStr
function calculateProbabilisticMetric(::Type{maskNumb} 
    ,G::Array{maskNumb, 3}
    ,T::Array{maskNumb, 3}
    ,TnTpFpFns::Vector{Float64} 
    ,isVariedSlice::Bool
    ,res::ResultMetrics) ::ResultMetrics


return setproperties(res, ( Kc=calculateCohenCappa(TnTpFpFns), )  ) 

end #calculateProbabilisticMetric


"""
calculate  Interclass correlaion
G,T - ground truth and evaluated masks - 3 dimensional arrays of supplied type-  maskNumb
isVariedSlice - true if slices are of variable thickness
res - ResultMetrics struct holding result of all metrics calculated for given run
return Interclass correlaion between G and T
"""
calculateInterClassCorrStr
function calculateInterClassCorr(::Type{maskNumb} 
    ,G::Array{maskNumb, 3}
    ,T::Array{maskNumb, 3}
    ,isVariedSlice::Bool) ::Float64

     mean_f = voxelprocesser->mean_f 
     mean_m = voxelprocesser->mean_m 
    int numberElements = std::min(voxelprocesser->numberElements_f, voxelprocesser->numberElements_m) 

     ssw = 0 
     ssb = 0 
     grandmean = (mean_f + mean_m)/2 
    for (int i = 0  i < numberElements  i++)
    {
         val_f = values_f[i] 
         val_m = values_m[i] 
         m = (val_f + val_m)/2 
        ssw += pow(val_f - m, 2) 
        ssw += pow(val_m - m, 2) 
        ssb += pow(m - grandmean, 2) 
    }
    ssw = ssw/numberElements 
    ssb = ssb/(numberElements-1) * 2 
     icc = (ssb - ssw)/(ssb + ssw) 
    return icc 

end #calculateInterClassCorr



"""
calculate Cohen Cappa  based on precalulated constants
TnTpFpFns - list of basic metrics - in the order [tn,tp,fp,fn ] 
return Cohen Cappa
"""
calculateCohenCappaStr
function calculateCohenCappa(TnTpFpFns::Vector{Float64} ) ::Float64
    tn = TnTpFpFns[1]
    tp = TnTpFpFns[2]
    fp = TnTpFpFns[3]
    fn = TnTpFpFns[4]
    agreement = tp + tn 
    chance_0 = (tn+fn)*(tn+fp) 
    chance_1 = (fp+tp)*(fn+tp) 
    chance = chance_0 + chance_1 
    sum = (tn + fn + fp + tp) 
    chance = chance/sum 
    return (agreement - chance)/(sum - chance) 
end #calculateVolumeMetric







end#ProbabilisticMetrics
