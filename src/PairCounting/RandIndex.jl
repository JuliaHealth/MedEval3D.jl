module RandIndex
    
using Main.BasicStructs, Parameters, Setfield


"""
calculate adjusted rand index based on precalulated constants
abcds - list of basic metrics - in the order [a,b,c,d ] 
res - ResultMetrics struct holding result of all metrics calculated for given run
here n = TP + FP + TN + FN - so all elements
"""
function calculateAdjustedRandIndex(tn,tp,fp, fn) ::Float64
	
    n= tn+tp+fp+fn

    coltot1 = tn + fp;
    coltot2 = fn + tp;
      rowtot1 = tn + fn;
      rowtot2 = fp + tp;
      nis = rowtot1*rowtot1 + rowtot2*rowtot2;  
      njs = coltot1*coltot1 + coltot2*coltot2;  
      s = tp*tp + tn*tn + fp*fp + fn*fn ;
    a = ( binomial(tn) + binomial(fn) + binomial(fp) + binomial(tp) )/2.0;
    b = (njs - s)/2.0;
    c = (nis - s)/2.0;
    d = ( n*n + s - nis - njs )/2.0; 
      t1=  (n * (n-1))/2.0 ;  

    x1 = a - ((a+c)*(a+b)/(a+b+c+d));

    x2 = ( (a+c) + (a+b))/2.0;

    x3 = ( (a+c)*(a+b))/(a+b+c+d);
    if(x2!=x3)
            return x1/(x2-x3)
    
    end
    return 0.0   
end #calculateAdjustedRandIndex

"""
helper function calculating binomial
"""
function binomial(val)
        return  val*(val-1)
end #binomial





end #RandIndex