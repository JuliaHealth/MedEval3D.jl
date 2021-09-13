
module LoadTestDataIntoJulia
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

subject_dirs = [subject for subject in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(subject) and os.path.basename(subject).startswith("Subject")]

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
    
    metrics = [pymMetr.DiceCoefficient()
                , pymMetr.HausdorffDistance(percentile=95, metric="HDRFDST95")
                , pymMetr.VolumeSimilarity()
                ,pymMetr.TruePositive()                ]
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


maxSlicesPerBlock,slicesPerBlockMatrix,numberOfBlocks = assignWorkToCooperativeBlocks(sizz[3], 3)

TpfpfnKernel.getTpfpfnData!(arrGold,arrAlgo,tp,tn,fp,fn
                            , intermediateResTp,intermediateResFp
                            ,intermediateResFn,sizz[1]*sizz[2],sizz[3]
                            ,UInt8(1)
                            ,IndexesArray,maxSlicesPerBlock,slicesPerBlockMatrix,numberOfBlocks)
tp[1]

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

tpTotalTrue = filter(pair->pair[2]== vec(goldS)[pair[1]] ==1 ,collect(enumerate(vec(segmAlgo))))|>length


arrOnesA = CUDA.ones(sizz);
arrOnesB = CUDA.ones(sizz);

goldBoolGPU,segmBoolGPU,tp,tn,fp,fn, tpArr,tnArr,fpArr, fnArr, blockNum , nx,ny,nz ,tpTotalTrue,tnTotalTrue,fpTotalTrue, fnTotalTrue ,tpPerSliceTrue,  tnPerSliceTrue,fpPerSliceTrue,fnPerSliceTrue ,flattG, flattSeg ,FlattGoldGPU,FlattSegGPU,intermediateResTp,intermediateResFp,intermediateResFn = getSmallTestBools();
IndexesArray= CUDA.zeros(Int32,10000000)

TpfpfnKernel.getTpfpfnData!(arrOnesA,arrOnesB,tp,tn,fp,fn, intermediateResTp,intermediateResFp,intermediateResFn,sizz[1]*sizz[2],sizz[3],Float32(1),IndexesArray)
tp[1]

length(arrOnesA) -sum(IndexesArray)


length(arrOnesA) - tp[1]


pixelNumberPerSlice = sizz[1]*sizz[2]
length(goldS)/pixelNumberPerSlice
loopNumb, indexCorr = getKernelContants(512,pixelNumberPerSlice)

blockIdx=1
threadIdx=512
correctedIdx = ((threadIdx-1)* indexCorr) +1
i =  (pixelNumberPerSlice*(blockIdx-1))+correctedIdx

pixelNumberPerSlice-correctedIdx

i/indexCorr

end #module