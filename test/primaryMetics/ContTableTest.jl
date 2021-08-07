using DrWatson
@quickactivate "Medical segmentation evaluation"
using Test
module ContTableTest

```@doc
testing caclulation of TP,FP,FN and TN
```
@testset "TpFpTnFnDiscrete " begin 
    @test    Main.GaussianPure.getOneNormDist(CartesianIndex(1,1,1),CartesianIndex(2,2,2)) == 3

end # getOneNormDist
    


end #ContTableTest

