
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

function getDataAndEvaluationFromPymia(examplemha)
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
    
    metrics = [pym.evaluation.metric.DiceCoefficient()
                , pym.evaluation.metric.HausdorffDistance(percentile=95
                , metric="HDRFDST95")
                , pym.evaluation.metric.VolumeSimilarity()
                , pym.evaluation.metric.TruePositive()
                ]
    evaluator = pym.evaluation.evaluator.SegmentationEvaluator(metrics, labels)
    
    evaluator.evaluate(prediction, ground_truth, getListOfMhdFromFolder(data_dir)[1][1])
    
    pym.evaluation.writer.ConsoleWriter().write(evaluator.results)
    
    goldS = np.array(sitk.GetArrayViewFromImage(ground_truth)) 
    segmAlgo = np.array(sitk.GetArrayViewFromImage(prediction))

    return (goldS,segmAlgo )
end#getDataAndEvaluationFromPymia

exampleFiles = getListOfExampleDatasFromFolder(data_dir)
examplemha = getListOfMhdFromFolder(data_dir)[2][3]


goldS,segmAlgo =getDataAndEvaluationFromPymia(examplemha);




size(pixels)
size(pixelsB)
#########myy
FlattB=pixels ;
FlattG=pixels ;

tpTotalTrue = filter(pair->pair[2]== FlattB[pair[1]] ==1 ,collect(enumerate(FlattG)))|>length





end #module