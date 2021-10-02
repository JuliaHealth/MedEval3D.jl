
#module LoadTestDataIntoJulia
using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\overLap\\MainOverlap.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\PairCounting\\RandIndex.jl")

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\Probabilistic\\ProbabilisticMetrics.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\InformationTheorhetic\\InformationTheorhetic.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\volume\\VolumeMetric.jl")

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernels\\TpfpfnKernel.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernels\\InterClassCorrKernel.jl")

using Main.BasicPreds, Main.CUDAGpuUtils 
using Main.MainOverlap, Main.TpfpfnKernel
using BenchmarkTools,StaticArrays

using Main.MainOverlap, Main.RandIndex , Main.ProbabilisticMetrics , Main.VolumeMetric ,Main.InformationTheorhetic


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
                
    #             ]
    metrics = [pymMetr.CohenKappaCoefficient()
                ,pymMetr.AdjustedRandIndex() 
                ,pymMetr.InterclassCorrelation()
                ]

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

arrGold = CuArray(vec(goldS))
arrAlgo = CuArray(vec(segmAlgo))
sizz= size(goldS)


goldBoolGPU,segmBoolGPU,tp,tn,fp,fn, tpArr,tnArr,fpArr, fnArr, blockNum , nx,ny,nz ,tpTotalTrue,tnTotalTrue,fpTotalTrue, fnTotalTrue ,tpPerSliceTrue,  tnPerSliceTrue,fpPerSliceTrue,fnPerSliceTrue ,flattG, flattSeg ,FlattGoldGPU,FlattSegGPU,intermediateResTp,intermediateResFp,intermediateResFn = getSmallTestBools();
IndexesArray= CUDA.zeros(Int32,10000000)
#TpfpfnKernel.getTpfpfnData!(arrGold,arrAlgo,tp,tn,fp,fn, intermediateResTp,intermediateResFp,intermediateResFn,sizz[1],sizz[1]*sizz[2],1,UInt8(1),IndexesArray)
using Main.BasicStructs
conf = ConfigurtationStruct(trues(12)...)
sliceMetricsTupl=(CUDA.zeros(sizz[3]),CUDA.zeros(sizz[3]),CUDA.zeros(sizz[3]),CUDA.zeros(sizz[3])
                            ,CUDA.zeros(sizz[3]),CUDA.zeros(sizz[3]),CUDA.zeros(sizz[3])
                            ,CUDA.zeros(sizz[3]),CUDA.zeros(sizz[3]),CUDA.zeros(sizz[3]),CUDA.zeros(sizz[3]) )#eleven entries

metricsTuplGlobal=  (CUDA.zeros(1),CUDA.zeros(1),CUDA.zeros(1),CUDA.zeros(1)
,CUDA.zeros(1),CUDA.zeros(1),CUDA.zeros(1)
,CUDA.zeros(1),CUDA.zeros(1),CUDA.zeros(1),CUDA.zeros(1) )#eleven entries

totalNumberOfVoxels=sizz[1]*sizz[2]*sizz[3]

argsB = TpfpfnKernel.getTpfpfnData!(arrGold,arrAlgo,tp,tn,fp,fn
                            ,sliceMetricsTupl
                            ,metricsTuplGlobal
                            ,sizz[1]*sizz[2]
                            ,sizz[3]
                            ,UInt8(1)
                            ,conf
                            ,totalNumberOfVoxels)
@test tp[1]==206422
#tn[1]==6684530
@test fp[1]==0
@test fn[1]==218185

#numbers below taken from pymia

@test isapprox(metricsTuplGlobal[4][1],0.654; atol = 0.1) #4) dice
@test isapprox(metricsTuplGlobal[5][1],0.486; atol = 0.1) #5) jaccard
@test isapprox(metricsTuplGlobal[6][1],0.000; atol = 0.1) #6) gce
@test isapprox(metricsTuplGlobal[7][1],0.618699; atol = 0.1) #7) randInd  - false
@test isapprox(metricsTuplGlobal[8][1],0.640; atol = 0.1) #8) cohen kappa - false
@test isapprox(metricsTuplGlobal[9][1],0.654; atol = 0.1) #9) volume metric
@test isapprox(metricsTuplGlobal[10][1],0.130; atol = 0.1) #10) mutual information
@test isapprox(metricsTuplGlobal[11][1],0.256; atol = 0.1) #11) variation of information


################## icc
sumOfGold= CuArray([0]);
sumOfSegm= CuArray([0]);

sswTotal= CUDA.zeros(1);
ssbTotal= CUDA.zeros(1);

iccPerSlice = CuArray(zeros(Float32,sizz[3]));
numberToLooFor= UInt8(1)
# arrGoldB = vec(CUDA.ones(UInt8,sizz));
# arrAlgoB =  vec(CUDA.ones(UInt8,sizz));
maxNumberOfBlocks = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)*1

globalICC= InterClassCorrKernel.calculateInterclassCorr(arrGold,arrAlgo
                                ,sizz
                                ,sumOfGold
                                ,sumOfSegm
                                ,sswTotal
                                ,ssbTotal
                                ,iccPerSlice
                                ,numberToLooFor
                                ,maxNumberOfBlocks)

@test isapprox(globalICC,0.6381813122385622; atol = 0.1)


goldS,segmAlgo 

goldSB = map(el->el== numberToLooFor ,vec(goldS));
segmAlgoB = map(el->el== numberToLooFor ,vec(segmAlgo));

mean_f = mean(goldSB)
mean_m = mean(segmAlgoB)
numberElements = length(segmAlgoB)

sumOfGold[1]/ (sizz[1]*sizz[2]*sizz[3]   )
# [ Info: grandMean 0.044381547296106404
# [ Info: numberOfVoxels 7109137
# 0.38722002506256104


		 ssw = 0
		 ssb = 0
		 grandmean = (mean_f + mean_m)/2
         icc=0
                 
for i in 1:(numberElements)
		
			 val_f = goldSB[i];
			 val_m = segmAlgoB[i];
			 m = (val_f + val_m)/2;
			ssw += (val_f - m)^2;
			ssw += (val_m - m)^2;
			ssb += (m - grandmean)^2;
end#for
ssw
ssb

isapprox(sswTotal[1] ,ssw; atol = 2)
isapprox(ssbTotal[1] ,ssb; atol = 2)



ssw = ssw/numberElements;
ssb = ssb/(numberElements-1) * 2;
icc = (ssb - ssw)/(ssb + ssw);

icc





BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds =60
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true


function toBench(FlattGoldGPU,FlattSegGPU,tp,tn,fp,fn)
    CUDA.@sync TpfpfnKernel.getTpfpfnData!(FlattGoldGPU,FlattSegGPU,tp,tn,fp,fn
                            ,sliceMetricsTupl
                            ,metricsTuplGlobal
                            ,sizz[1]*sizz[2]
                            ,sizz[3]
                            ,UInt8(1)
                            ,conf
                            ,totalNumberOfVoxels)
                        end

bb2 = @benchmark toBench(FlattGoldGPU,FlattSegGPU,tp,tn,fp,fn)  setup=(goldBoolGPU,segmBoolGPU,tp,tn,fp,fn, tpArr,tnArr,fpArr, fnArr, blockNum , nx,ny,nz ,tpTotalTrue,tnTotalTrue,fpTotalTrue, fnTotalTrue ,tpPerSliceTrue,  tnPerSliceTrue,fpPerSliceTrue,fnPerSliceTrue ,flattG, flattSeg ,FlattGoldGPU,FlattSegGPU,intermediateResTp,intermediateResFp,intermediateResFn = getSmallTestBools())


CUDA.@profile begin
    TpfpfnKernel.getTpfpfnData!(arrGold,arrAlgo,tp,tn,fp,fn
    , intermediateResTp,intermediateResFp
    ,intermediateResFn,sizz[1]*sizz[2],sizz[3]
    ,UInt8(1)
    ,IndexesArray)
end


## stored indexes  and now we are inspecting it 
diff = Int64(round(length(goldS)- maximum(IndexesArray)))



diff/pixelNumberPerSlice
cpuZZ = zeros(Int32,10000000)
copyto!(cpuZZ, zz)
zz = filter(it->it>0 ,cpuZZ)


diffrenceOfIndexes = filter(pair->pair[1]!=pair[2], collect(enumerate(zz))  )
length(zz) - length(goldS)

sort(zz)

pixelNumberPerSlice-512*77

# Subject_2  GREYMATTER   0.298  10.863     74392.000   0.298
# Subject_2  WHITEMATTER  0.654  6.000      206422.000  0.654

#tpTotalTrue = filter(pair->pair[2]== vec(goldS)[pair[1]] ==1 ,collect(enumerate(vec(segmAlgo))))|>length


arrOnesA = CUDA.ones(sizz);
arrOnesB = CUDA.ones(sizz);

goldBoolGPU,segmBoolGPU,tp,tn,fp,fn, tpArr,tnArr,fpArr, fnArr, blockNum , nx,ny,nz ,tpTotalTrue,tnTotalTrue,fpTotalTrue, fnTotalTrue ,tpPerSliceTrue,  tnPerSliceTrue,fpPerSliceTrue,fnPerSliceTrue ,flattG, flattSeg ,FlattGoldGPU,FlattSegGPU,intermediateResTp,intermediateResFp,intermediateResFn = getSmallTestBools();
IndexesArray= CUDA.zeros(Int32,10000000)

TpfpfnKernel.getTpfpfnData!(arrOnesA,arrOnesB,tp,tn,fp,fn, intermediateResTp,intermediateResFp,intermediateResFn,sizz[1]*sizz[2],sizz[3],Float32(1),IndexesArray)
# sum(IndexesArray)
tp[1]
tp[1] -length(arrOnesA)
# length(arrOnesA) -sum(IndexesArray)
#length(arrOnesA) - tp[1]


tp = 5
fp = 5
fn = 5
tn = 5


using Main.RandIndex

MainOverlap.dice(tp,fp, fn)
MainOverlap.jaccard(tp,fp, fn)
MainOverlap.gce(tn,tp,fp, fn)
RandIndex.calculateAdjustedRandIndex(tn,tp,fp, fn)
ProbabilisticMetrics.calculateCohenCappa(tn,tp,fp, fn )
VolumeMetric.getVolumMetric(tp,fp, fn )
InformationTheorhetic.mutualInformationMetr(tn,tp,fp, fn)
InformationTheorhetic.variationOfInformation(tn,tp,fp, fn)





rand(1,3,7)[19]


end #module

