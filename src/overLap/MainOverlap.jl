
"""
calculates main overlap metrics
based on https://github.com/Visceral-Project/EvaluateSegmentation/blob/master/source/DiceCoefficientMetric.h
https://github.com/Visceral-Project/EvaluateSegmentation/blob/master/source/JaccardCoefficientMetric.h
https://github.com/Visceral-Project/EvaluateSegmentation/blob/master/source/GlobalConsistencyError.h
"""
module MainOverlap
using Main.BasicStructs, Parameters, Setfield

export calculateBAsicOverlap, dice, jaccard, gce

"""
calculate dice coefficient based on precalulated constants
TnTpFpFns - list of basic metrics - in the order [tn,tp,fp,fn ] 
res - ResultMetrics struct holding result of all metrics calculated for given run
return modified ResultMetrics with added Dice, Jaccard and global consistency error coefficients 
"""
function calculateBAsicOverlap(TnTpFpFns::Vector{Float64} , res::ResultMetrics) ::ResultMetrics
    tn = TnTpFpFns[1]
    tp = TnTpFpFns[2]
    fp = TnTpFpFns[3]
    fn = TnTpFpFns[4]

    return setproperties(res, (dice=  dice(tn,tp,fp) ,jaccard=  jaccard(tn,tp,fp) ,gce =  gce(tn,tp,fp,fn)  )) 

end #calculateDice

```@doc
Calculates Dice Coefficient  on the basis of true negative, true positive, false positive and false negatives 
 ```
function dice(tp,fp, fn)::Float64 
 return  2*tp/(2*tp + fp + fn)
end#dice


```@doc
Calculates Jaccard Coefficient  on the basis of true negative, true positive, false positive and false negatives 
```
function jaccard(tp,fp, fn)::Float64 
 return  tp/(tp + fp + fn)
end#jaccard


```@doc
Calculates Global Consistency error on the basis of true negative, true positive, false positive and false negatives 
```
function gce(tn,tp,fp, fn)::Float64 
    n = tn+fp+fn+tp;
    e1 = ( fn*(fn+ 2*tp)/(tp+fn) + fp*(fp + 2*tn)/(tn+fp) )/n;
    e2 = ( fp*(fp+2*tp)/(tp+fp) + fn*(fn + 2*tn)/(tn+fn) )/n;
    return min(e1, e2);
end#dice


end #MainOverlap