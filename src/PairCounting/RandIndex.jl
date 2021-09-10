using DrWatson
@quickactivate "Medical segmentation evaluation"

module RandIndex
    
using Main.BasicStructs, Parameters, Setfield


"""
calculate adjusted rand index based on precalulated constants
abcds - list of basic metrics - in the order [a,b,c,d ] 
res - ResultMetrics struct holding result of all metrics calculated for given run
return modified ResultMetrics with added adjusted rand index
"""

function calculateAdjustedRandIndex(abcds::Vector{Float64} , res::ResultMetrics) ::ResultMetrics
    a = abcds[1]
    b = abcds[2]
    c = abcds[3]
    d = abcds[4]

    x1 = a - ((a+c)*(a+b)/(a+b+c+d));
    x2 = ( (a+c) + (a+b))/2.0;
    x3 = ( (a+c)*(a+b))/(a+b+c+d);

    if(x2!=x3)
        setproperties(res, (randInd= x1/(x2-x3) )) 
    else
        return res;
    end#if


return setproperties(res, (vol=  1- abs(fn-fp)/(2*tp + fn + fp)  )) 

end #calculateAdjustedRandIndex





end #RandIndex