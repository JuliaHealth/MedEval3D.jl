using DrWatson
@quickactivate "Medical segmentation evaluation"

module RandIndex
    
using Main.BasicStructs, Parameters, Setfield


"""
calculate adjusted rand index based on precalulated constants
abcds - list of basic metrics - in the order [a,b,c,d ] 
res - ResultMetrics struct holding result of all metrics calculated for given run
here n = TP + FP + TN + FN - so all elements
"""
function calculateAdjustedRandIndex(tn,tp,fp, fn) ::ResultMetrics
	
    n= tn+tp+fp+fn

	rowtot1 = tn + fn;
	rowtot2 = fp + tp;
    nis = rowtot1*rowtot1 + rowtot2*rowtot2;  

	coltot1 = tn + fp;
	coltot2 = fn + tp;
    njs = coltot1*coltot1 + coltot2*coltot2;  


    s = tp*tp + tn*tn + fp*fp + fn*fn ;    

    a = ( binomial(tn) + binomial(fn) + binomial(fp) + binomial(tp) )/2.0;

    b = (njs - s)/2.0;

    c = (nis - s)/2.0;

    d = ( n*n + s - nis - njs )/2.0;     

    x1 = a - ((a+c)*(a+b)/(a+b+c+d));

    x2 = ( (a+c) + (a+b))/2.0;

    x3 = ( (a+c)*(a+b))/(a+b+c+d);




return setproperties(res, (vol=  1- abs(fn-fp)/(2*tp + fn + fp)  )) 

end #calculateAdjustedRandIndex

"""
helper function calculating binomial
"""
function binomial(val::Float64)::Float64
    return  val*(val-1)
end #binomial





end #RandIndex