
#module LoadTestDataIntoJulia
using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils, ..BasicStructs
using ..ResultListUtils, ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils, ..ScanForDuplicates
using ..MainOverlap, ..RandIndex , ..ProbabilisticMetrics , ..VolumeMetric ,..InformationTheorhetic
using ..CUDAAtomicUtils, ..TpfpfnKernel, ..InterClassCorrKernel,..MeansMahalinobis
using ..MainAbstractions

using Conda
using PyCall
using Pkg

Conda.pip_interop(true)
Conda.pip("install", "SimpleITK")
Conda.pip("install", "pandas")
Conda.pip("install", "pymia")

sitk = pyimport("SimpleITK")
pym = pyimport("pymia")
pymMetr = pyimport("pymia.evaluation.metric")
pymEval = pyimport("pymia.evaluation.evaluator")
pymWrite = pyimport("pymia.evaluation.writer")
np= pyimport("numpy")

data_dir = "C:\\GitHub\\GitHub\\NuclearMedEval\\test\\data\\exampleForTestsData"
result_file = "C:\\GitHub\\GitHub\\NuclearMedEval\\test\\data\\pymiaOutput\\results.csv"
result_summary_file = "C:\\GitHub\\GitHub\\NuclearMedEval\\test\\data\\pymiaOutput\\results_summary.csv"


#given directory it gives all mhd file names concateneted with path - to get full file path and second in subarray will be file name
function getListOfExampleDatasFromFolder(folderPath::String) ::Vector{Vector{AbstractString}}
    return readdir(folderPath) |>
    (arr)-> filter((str)-> occursin("Subject",str), arr) |>
    (arr)-> map(str-> [split(str,".")[1], joinpath(folderPath,str), "$(joinpath(joinpath(folderPath,str),str))_GT.mha" ],arr)
end

function getDataAndEvaluationFromPymia(examplemhaDat)
    examplemha= examplemhaDat[3]
    ground_truth = sitk.ReadImage(examplemha)
    prediction = ground_truth
   
   py"""
    import SimpleITK as sitk
    def getPred(prediction, numb):
        return sitk.BinaryErode(prediction, [1,1,1], sitk.sitkBall, 0,numb)
    """
    prediction= py"getPred"(prediction,1)
    prediction= py"getPred"(prediction,2)
    
    labels = Dict([(1,"WHITEMATTER"),(2,"GREYMATTER") ])
    
    # metrics = [pymMetr.DiceCoefficient()
    #             , pymMetr.DiceCoefficient()
    #             , pymMetr.JaccardCoefficient()
    #             , pymMetr.GlobalConsistencyError()
    #             , pymMetr.AdjustedRandIndex()
    #             , pymMetr.CohenKappaCoefficient()
    #             , pymMetr.MutualInformation()
    #             , pymMetr.VariationOfInformation()
    #             , pymMetr.VolumeSimilarity()
    #             ,pymMetr.TruePositive()   
    #             ,pymMetr.TrueNegative()   
    #             ,pymMetr.FalsePositive()   
    #             ,pymMetr.FalsePositive()   
    # pymMetr.CohenKappaCoefficient()
    #             ,pymMetr.AdjustedRandIndex() 
    #             ,pymMetr.InterclassCorrelation()
                
    #             ]
    metrics = [pymMetr.MahalanobisDistance()                ]

    evaluator = pymEval.SegmentationEvaluator(metrics, labels)
    
    evaluator.evaluate(prediction, ground_truth, examplemhaDat[1])
    
    pymWrite.ConsoleWriter().write(evaluator.results)
    
    goldS = np.array(sitk.GetArrayViewFromImage(ground_truth)) 
    segmAlgo = np.array(sitk.GetArrayViewFromImage(prediction))

    return (goldS,segmAlgo )
end#getDataAndEvaluationFromPymia

exampleFiles = getListOfExampleDatasFromFolder(data_dir)
examplemhaDat = exampleFiles[2]


goldS,segmAlgo =getDataAndEvaluationFromPymia(examplemhaDat);

arrGold = CuArray(goldS)
arrAlgo = CuArray(segmAlgo)
sizz= size(goldS)
##### load tests ...
conf = ConfigurtationStruct(false,trues(11)...)
numberToLooFor = UInt8(1)

preparedDict=MainAbstractions.prepareMetrics(conf)
res = calcMetricGlobal(preparedDict,conf,arrGold,arrAlgo,numberToLooFor)
res
#numbers below taken from pymia

@test isapprox(res.dice,0.654; atol = 0.1) #4) dice
@test isapprox(res.jaccard,0.486; atol = 0.1) #5) jaccard
@test isapprox(res.gce ,0.000; atol = 0.1) #6) gce
@test isapprox(res.randInd,0.618699; atol = 0.1) #7) randInd  
@test isapprox(res.kc,0.640; atol = 0.1) #8) cohen kappa 
@test isapprox(res.vol,0.654; atol = 0.1) #9) volume metric
@test isapprox(res.mi,0.130; atol = 0.1) #10) mutual information
@test isapprox(res.vi,0.256; atol = 0.1) #11) variation of information
@test isapprox(res.ic,0.6381813122385622; atol = 0.1)# interclas correlation
@test isapprox(res.md,0.08;atol = 0.1 ) #Mahalinobis


# ################# icc
# argsMain, threads,blocks, totalNumbOfVoxels=InterClassCorrKernel.prepareInterClassCorrKernel(arrGold ,arrAlgo,numberToLooFor)
# globalICC= InterClassCorrKernel.calculateInterclassCorr(arrGold,arrAlgo,threads,blocks,argsMain)



# # ################ Mahalinobis 
# using ..MeansMahalinobis
# args,threads ,blocks= MeansMahalinobis.prepareMahalinobisKernel()
# mahalanobisResGlob=  MeansMahalinobis.calculateMalahlinobisDistance(arrGold,arrAlgo,args,threads ,blocks,1)
# goldS3d= CuArray(goldS);
# segmS3d= CuArray(segmAlgo);
# #we will fill it after we work with launch configuration
# loopXdim = UInt32(1);loopYdim = UInt32(1) ;loopZdim = UInt32(1) ;
# sizz = size(goldS3d);maxX = UInt32(sizz[1]);maxY = UInt32(sizz[2]);maxZ = UInt32(sizz[3])
# #gold
# totalXGold= CuArray([0.0]);
# totalYGold= CuArray([0.0]);
# totalZGold= CuArray([0.0]);
# totalCountGold= CuArray([0]);
# #segm
# totalXSegm= CuArray([0.0]);
# totalYSegm= CuArray([0.0]);
# totalZSegm= CuArray([0.0]);

# varianceXGlobalGold,covarianceXYGlobalGold,covarianceXZGlobalGold,varianceYGlobalGold,covarianceYZGlobalGold,varianceZGlobalGold= CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]);
# varianceXGlobalSegm,covarianceXYGlobalSegm,covarianceXZGlobalSegm,varianceYGlobalSegm,covarianceYZGlobalSegm,varianceZGlobalSegm= CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]);

# totalCountSegm= CuArray([0]);
# totalCountGold
# #countPerZGold= CUDA.zeros(Float32,sizz[3]+1);
# #countPerZSegm= CUDA.zeros(Float32,sizz[3]+1);
# countPerZGold= CUDA.zeros(Float32,500);
# countPerZSegm= CUDA.zeros(Float32,500);

# #covariancesSliceWise= CUDA.zeros(Float32,12,sizz[3]+1);
# covariancesSliceWiseGold= CUDA.zeros(Float32,6,500);
# covariancesSliceWiseSegm= CUDA.zeros(Float32,6,500);
# covarianceGlobal= CUDA.zeros(Float32,12,1);

# mahalanobisResGlobal= CUDA.zeros(1);
# mahalanobisResSliceWise= CUDA.zeros(500);
# #mahalanobisResSliceWise= CUDA.zeros(sizz[3]);

# args = (goldS3d,segmS3d,numberToLooFor
# ,loopYdim,loopXdim,loopZdim
# ,(maxX, maxY,maxZ)
# ,totalXGold,totalYGold,totalZGold,totalCountGold
# ,totalXSegm,totalYSegm,totalZSegm,totalCountSegm,countPerZGold
# , countPerZSegm,covariancesSliceWiseGold, covariancesSliceWiseSegm,
# varianceXGlobalGold,covarianceXYGlobalGold,covarianceXZGlobalGold,varianceYGlobalGold,covarianceYZGlobalGold,varianceZGlobalGold
#     ,varianceXGlobalSegm,covarianceXYGlobalSegm,covarianceXZGlobalSegm,varianceYGlobalSegm,covarianceYZGlobalSegm,varianceZGlobalSegm
#     ,mahalanobisResGlobal, mahalanobisResSliceWise)




#     # calculate the amount of dynamic shared memory for a 2D block size
#     get_shmem(threads) = (sizeof(UInt32)*3*4)
    
#     function get_threads(threads)
#         threads_x = 32
#         threads_y = cld(threads,threads_x )
#         return (threads_x, threads_y)
#     end

#     kernel = @cuda launch=false MeansMahalinobis.meansMahalinobisKernel(args...)
   
#     config = launch_configuration(kernel.fun, shmem=threads->get_shmem(get_threads(threads)))

#    # convert to 2D block size and figure out appropriate grid size
#     threads = get_threads(config.threads)
#     blocks = UInt32(config.blocks)
#     loopXdim = UInt32(cld(maxX, threads[1]))
#     loopYdim = UInt32(cld(maxY, threads[2])) 
#     loopZdim = UInt32(cld(maxZ,blocks )) 

# #covariancesSliceWise= CUDA.zeros(Float32,12,sizz[3]+1);
# covariancesSliceWiseGold= CUDA.zeros(Float32,6,500);
# covariancesSliceWiseSegm= CUDA.zeros(Float32,6,500);

# #gold
# totalXGold= CuArray([0.0]);
# totalYGold= CuArray([0.0]);
# totalZGold= CuArray([0.0]);
# totalCountGold= CuArray([0]);
# #segm
# totalXSegm= CuArray([0.0]);
# totalYSegm= CuArray([0.0]);
# totalZSegm= CuArray([0.0]);
# totalCountSegm= CuArray([0]);



# varianceXGlobalGold,covarianceXYGlobalGold,covarianceXZGlobalGold,varianceYGlobalGold,covarianceYZGlobalGold,varianceZGlobalGold= CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]);
# varianceXGlobalSegm,covarianceXYGlobalSegm,covarianceXZGlobalSegm,varianceYGlobalSegm,covarianceYZGlobalSegm,varianceZGlobalSegm= CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]);



# args = (goldS3d,segmS3d,numberToLooFor
# ,loopYdim,loopXdim,loopZdim
# ,(maxX, maxY,maxZ)
# ,totalXGold,totalYGold,totalZGold,totalCountGold
# ,totalXSegm,totalYSegm,totalZSegm,totalCountSegm,countPerZGold
# , countPerZSegm,covariancesSliceWiseGold, covariancesSliceWiseSegm,
# varianceXGlobalGold,covarianceXYGlobalGold,covarianceXZGlobalGold,varianceYGlobalGold,covarianceYZGlobalGold,varianceZGlobalGold
#     ,varianceXGlobalSegm,covarianceXYGlobalSegm,covarianceXZGlobalSegm,varianceYGlobalSegm,covarianceYZGlobalSegm,varianceZGlobalSegm
#     ,mahalanobisResGlobal, mahalanobisResSliceWise)


#     @cuda cooperative=true threads=threads blocks=blocks MeansMahalinobis.meansMahalinobisKernel(args...)
    
#     @test isapprox(mahalanobisResGlobal[1],0.08;atol = 0.1 )











# ####################### some benchmarks 


# BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
# BenchmarkTools.DEFAULT_PARAMETERS.seconds =60
# BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true


# function toBench(FlattGoldGPU,FlattSegGPU,tp,tn,fp,fn)
#     CUDA.@sync TpfpfnKernel.getTpfpfnData!(FlattGoldGPU,FlattSegGPU,tp,tn,fp,fn
#                             ,sliceMetricsTupl
#                             ,metricsTuplGlobal
#                             ,sizz[1]*sizz[2]
#                             ,sizz[3]
#                             ,UInt8(1)
#                             ,conf
#                             ,totalNumberOfVoxels)
#                         end

# bb2 = @benchmark toBench(FlattGoldGPU,FlattSegGPU,tp,tn,fp,fn)  setup=(goldBoolGPU,segmBoolGPU,tp,tn,fp,fn, tpArr,tnArr,fpArr, fnArr, blockNum , nx,ny,nz ,tpTotalTrue,tnTotalTrue,fpTotalTrue, fnTotalTrue ,tpPerSliceTrue,  tnPerSliceTrue,fpPerSliceTrue,fnPerSliceTrue ,flattG, flattSeg ,FlattGoldGPU,FlattSegGPU,intermediateResTp,intermediateResFp,intermediateResFn = getSmallTestBools())


# CUDA.@profile begin
#     TpfpfnKernel.getTpfpfnData!(arrGold,arrAlgo,tp,tn,fp,fn
#     , intermediateResTp,intermediateResFp
#     ,intermediateResFn,sizz[1]*sizz[2],sizz[3]
#     ,UInt8(1)
#     ,IndexesArray)
# end


# ## stored indexes  and now we are inspecting it 
# diff = Int64(round(length(goldS)- maximum(IndexesArray)))



# diff/pixelNumberPerSlice
# cpuZZ = zeros(Int32,10000000)
# copyto!(cpuZZ, zz)
# zz = filter(it->it>0 ,cpuZZ)


# diffrenceOfIndexes = filter(pair->pair[1]!=pair[2], collect(enumerate(zz))  )
# length(zz) - length(goldS)

# sort(zz)

# pixelNumberPerSlice-512*77

# # Subject_2  GREYMATTER   0.298  10.863     74392.000   0.298
# # Subject_2  WHITEMATTER  0.654  6.000      206422.000  0.654

# #tpTotalTrue = filter(pair->pair[2]== vec(goldS)[pair[1]] ==1 ,collect(enumerate(vec(segmAlgo))))|>length


# arrOnesA = CUDA.ones(sizz);
# arrOnesB = CUDA.ones(sizz);

# goldBoolGPU,segmBoolGPU,tp,tn,fp,fn, tpArr,tnArr,fpArr, fnArr, blockNum , nx,ny,nz ,tpTotalTrue,tnTotalTrue,fpTotalTrue, fnTotalTrue ,tpPerSliceTrue,  tnPerSliceTrue,fpPerSliceTrue,fnPerSliceTrue ,flattG, flattSeg ,FlattGoldGPU,FlattSegGPU,intermediateResTp,intermediateResFp,intermediateResFn = getSmallTestBools();
# IndexesArray= CUDA.zeros(Int32,10000000)

# TpfpfnKernel.getTpfpfnData!(arrOnesA,arrOnesB,tp,tn,fp,fn, intermediateResTp,intermediateResFp,intermediateResFn,sizz[1]*sizz[2],sizz[3],Float32(1),IndexesArray)
# # sum(IndexesArray)
# tp[1]
# tp[1] -length(arrOnesA)
# # length(arrOnesA) -sum(IndexesArray)
# #length(arrOnesA) - tp[1]


# tp = 5
# fp = 5
# fn = 5
# tn = 5


# using ..RandIndex

# MainOverlap.dice(tp,fp, fn)
# MainOverlap.jaccard(tp,fp, fn)
# MainOverlap.gce(tn,tp,fp, fn)
# RandIndex.calculateAdjustedRandIndex(tn,tp,fp, fn)
# ProbabilisticMetrics.calculateCohenCappa(tn,tp,fp, fn )
# VolumeMetric.getVolumMetric(tp,fp, fn )
# InformationTheorhetic.mutualInformationMetr(tn,tp,fp, fn)
# InformationTheorhetic.variationOfInformation(tn,tp,fp, fn)





# rand(1,3,7)[19]


# end #module

