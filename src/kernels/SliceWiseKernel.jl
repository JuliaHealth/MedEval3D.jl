"""
given appropriate constants related to slice like rate of true pasitives, fals positive, false negative
and false negative
it calculates wanted  segmentation metrics
in order to reduce thread divergence and enable easy omitting this step - in case we want only global statistics 
we will  calculate it on  in a seperate kernel than the data about true positives ... is calculated
"""
function calculateBasicMetricsPerSlice()

    dice, jaccard, gce
end
