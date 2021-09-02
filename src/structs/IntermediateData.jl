"""
Storing the intermediate results  - like true positives
... and other constants that will be used to given metricks  
"""
module  IntermediateData

using Parameters


tn += min(y1,y2)
tp += min(x1,x2)
fn += x1>x2 ? x1-x2 : 0
fp += x2>x1 ? x2-x1 : 0

end#IntermediateData