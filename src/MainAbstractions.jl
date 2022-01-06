"""
making it possible to invoke metrics in simple way
"""
module MainAbstractions
using CUDA 
using Revise, Parameters, Logging, Test
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils, ..BasicStructs
using ..ResultListUtils, ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils, ..ScanForDuplicates
using ..MainOverlap, ..RandIndex , ..ProbabilisticMetrics , ..VolumeMetric ,..InformationTheorhetic
using ..CUDAAtomicUtils, ..TpfpfnKernel, ..InterClassCorrKernel,..MeansMahalinobis
export prepareMetrics, calcMetricGlobal
"""
function created to prepare  kernel launch - required data is summarized in the ConfigurtationStruct
    confStruct- specified by the user - where is the summary of what metrics should be measured
    numberToLookFor - the number representing label which will be checked in the target arrays
"""
function prepareMetrics(confStruct::ConfigurtationStruct)

    dict = Dict()
    
    isAnyConfusionMatrixMetric = (confStruct.dice || confStruct.jaccard|| confStruct.gce|| confStruct.dice|| confStruct.vol|| confStruct.randInd|| confStruct.kc|| confStruct.mi || confStruct.vi)
    
    if(!confStruct.sliceWiseMetrics &&   isAnyConfusionMatrixMetric)
        dict[1]= prepareForconfusionTableMetricsNoSliceWise(confStruct)
    end    
    if(confStruct.ic)
        dict[2]=InterClassCorrKernel.prepareInterClassCorrKernel()
    end    
    if(confStruct.md)
        dict[3]= MeansMahalinobis.prepareMahalinobisKernel()
    end  

    return dict
end    

"""
executing metrics calculations
    preparedDict - dictionary created in the preparatory step
    confStruct- specified by the user - where is the summary of what metrics should be measured
    goldGPU,segmGPU - 3 dimensional arrays representing gold standard and output of our algorithm
    numberToLookFor - number representing the label of intrest in the arrays
"""
function calcMetricGlobal(preparedDict, confStruct,goldGPU,segmGPU, numberToLookFor)::ResultMetrics
    res = ResultMetrics()

    numberToLooFor=numberToLookFor
    isAnyConfusionMatrixMetric = (confStruct.dice || confStruct.jaccard|| confStruct.gce|| confStruct.dice|| confStruct.vol|| confStruct.randInd|| confStruct.kc|| confStruct.mi || confStruct.vi)   
    
    if(isAnyConfusionMatrixMetric)
        argss,threads,blocks,metricsTuplGlobal=preparedDict[1] 
        TpfpfnKernel.getTpfpfnData!(goldGPU,segmGPU   ,argss,threads,blocks,metricsTuplGlobal,numberToLooFor,confStruct)  
        res.dice = metricsTuplGlobal[4][1]
        res.jaccard = metricsTuplGlobal[5][1]
        res.gce = metricsTuplGlobal[6][1]
        res.randInd = metricsTuplGlobal[7][1]
        res.kc = metricsTuplGlobal[8][1]
        res.vol = metricsTuplGlobal[9][1]
        res.mi = metricsTuplGlobal[10][1]
        res.vi = metricsTuplGlobal[11][1]
    end    
    if(confStruct.ic)
        argsMain, threads,blocks, totalNumbOfVoxels=preparedDict[2]
        res.ic = InterClassCorrKernel.calculateInterclassCorr(goldGPU,segmGPU,threads,blocks,argsMain,numberToLooFor)
    end
    if(confStruct.md)
        argsMahal,threadsMahal ,blocksMahal= preparedDict[3]
        res.md = MeansMahalinobis.calculateMalahlinobisDistance(goldGPU,segmGPU,argsMahal,threadsMahal ,blocksMahal,numberToLooFor)
    end  

    return res
    end#calcMetricGlobal

end