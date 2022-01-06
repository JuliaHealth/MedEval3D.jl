# MedEval3D

Project with set of CUDA accelerated  medical segmentation metrics - currently in active development.

Package is not yet updated to official repository yet metrics mentioned below are usable - expected inputs are 3 dimensional arrays - 

```
# 3 dimensional arrays representing gold standard an output of our algorithm
arrGold = CuArray((not_translated))
arrAlgo = CuArray((translated))

#Dice
conf = ConfigurtationStruct(dice=true)
#number that is encoding the label of intrest in given arrays
numberToLooFor = UInt8(1)
    argss,threads,blocks,metricsTuplGlobal= TpfpfnKernel.prepareForconfusionTableMetrics(arrGold    , arrAlgo    ,numberToLooFor  ,conf)
#preparation - we precompute all that is possible increases performance given all of the arrays we will check have the same dimensions
argsB = TpfpfnKernel.getTpfpfnData!(arrGold ,arrAlgo   ,argss,threads,blocks,metricsTuplGlobal) 

#CUDA execution of a metric
TpfpfnKernel.getTpfpfnData!(arrGold ,arrAlgo   ,argsB,threads,blocks,metricsTuplGlobal)# setup = (arrGold ,arrAlgo   ,args,threads,blocks,metricsTuplGlobal = clearFunction()   )
 result for Dice is in metricsTuplGlobal[4][1]
 ```
For executing othe metrics we need to specify appropriate ConfigurtationStruct, we can do multiple metrics at once and it will run in similar time as single metric (most of the computations are reused)

To define the merics we are intrested in define ConfigurtationStruct appropriately

```
ConfigurtationStruct
    dice::Bool = false #dice coefficient
    jaccard::Bool = false #jaccard coefficient
    gce::Bool = false #global consistency error
    vol::Bool = false# Volume metric
    randInd::Bool= false # Rand Index 
    ic::Bool= false # interclass correlation
    kc::Bool= false # Kohen Cappa
    mi::Bool= false # mutual information
    vi::Bool= false # variation Of Information
```
in order to get results  we need to access the tuple with results as seen below
```
metricsTuplGlobal[4][1] #dice coefficient
metricsTuplGlobal[5][1] #jaccard coefficient
metricsTuplGlobal[6][1] #global consistency error
metricsTuplGlobal[7][1] # Volume metric
metricsTuplGlobal[8][1] # Rand Index 
metricsTuplGlobal[9][1]  # interclass correlation
metricsTuplGlobal[10][1] # Kohen Cappa
metricsTuplGlobal[11][1] # mutual information
metricsTuplGlobal[12][1] # variation Of Information



```
