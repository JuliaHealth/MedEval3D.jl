"""
given appropriate constants related to slice like rate of true pasitives, fals positive, false negative
and false negative
it calculates wanted  segmentation metrics
in order to reduce thread divergence and enable easy omitting this step - in case we want only global statistics 
we will  calculate it on  in a seperate kernel than the data about true positives ... is calculated
"""
function calculateBasicMetricsPerSlice(tp,tn,fp,fn
                                        ,intermediateResTp
                                        ,intermediateResFp
                                        ,intermediateResFn
                                        ,)

    dice, jaccard, gce
end

MainOverlap.dice(tp,fp, fn)
MainOverlap.jaccard(tp,fp, fn)
MainOverlap.gce(tp,fp, fn)
RandIndex.calculateAdjustedRandIndex(tn,tp,fp, fn)
ProbabilisticMetrics.calculateCohenCappa(tp,fp, fn )
VolumeMetric.getVolumMetric(tp,fp, fn )
InformationTheorhetic.mutualInformationMetr(tn,tp,fp, fn)
InformationTheorhetic.variationOfInformation(tn,tp,fp, fn)